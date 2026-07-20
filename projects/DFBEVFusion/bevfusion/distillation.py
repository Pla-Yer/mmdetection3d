"""Training-only LiDAR knowledge distillation components."""

from copy import deepcopy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import MessageHub
from torch import Tensor

from mmdet3d.models import (Base3DDetector, draw_heatmap_gaussian,
                            gaussian_radius)
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample

from .bevfusion import DFBEVFusion


@MODELS.register_module()
class DistillBEVFusionTeacher(DFBEVFusion):
    """Original sparse-voxel BEVFusion under a collision-free registry name."""

    def extract_pts_feat(self, batch_inputs_dict) -> Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        return self.pts_middle_encoder(feats, coords, batch_size)


class _WeightedDistillLoss(nn.Module):
    def __init__(self, loss_weight=1.0, enabled=True,
                 confidence_threshold=0.1, background_weight=0.1,
                 spatial_temperature=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.enabled = bool(enabled)
        self.confidence_threshold = float(confidence_threshold)
        self.background_weight = float(background_weight)
        self.spatial_temperature = float(spatial_temperature)
        if not 0 <= self.confidence_threshold < 1:
            raise ValueError('confidence_threshold must be in [0, 1)')
        if not 0 <= self.background_weight <= 1:
            raise ValueError('background_weight must be in [0, 1]')
        if self.spatial_temperature <= 0:
            raise ValueError('spatial_temperature must be positive')

    def spatial_weight(self, teacher_heatmap, dtype, temperature=None):
        """Return calibration-independent soft foreground weights.

        TransFusion focal logits have very small absolute probabilities, so an
        absolute 0.1 cutoff can silently become a uniform background map.  Use
        per-sample relative confidence and retain ``confidence_threshold`` as
        a relative salience floor instead.
        """
        temperature = (self.spatial_temperature if temperature is None
                       else float(temperature))
        confidence = (teacher_heatmap.detach().float() / temperature).sigmoid()
        confidence = confidence.amax(dim=1, keepdim=True)
        minimum = confidence.amin(dim=(-2, -1), keepdim=True)
        maximum = confidence.amax(dim=(-2, -1), keepdim=True)
        relative = (confidence - minimum) / (maximum - minimum).clamp_min(1e-6)
        salience = ((relative - self.confidence_threshold) /
                    (1.0 - self.confidence_threshold)).clamp_(0, 1)
        weight = self.background_weight + (1.0 - self.background_weight) * salience
        return weight.to(dtype=dtype)


@MODELS.register_module()
class BEVFeatureDistillLoss(_WeightedDistillLoss):
    def __init__(self, norm_eps=1e-4, energy_threshold=0.01,
                 energy_power=1.0, **kwargs):
        super().__init__(**kwargs)
        self.norm_eps = float(norm_eps)
        self.energy_threshold = float(energy_threshold)
        self.energy_power = float(energy_power)
        if not 0 <= self.energy_threshold < 1:
            raise ValueError('energy_threshold must be in [0, 1)')
        if self.energy_power <= 0:
            raise ValueError('energy_power must be positive')

    def _prepare_features(self, student, teacher):
        student = student.float()
        teacher = teacher.detach().float()
        teacher_energy = teacher.norm(dim=1, keepdim=True)
        peak = teacher_energy.amax(dim=(-2, -1), keepdim=True)
        relative_energy = teacher_energy / peak.clamp_min(self.norm_eps)
        energy_weight = torch.where(
            relative_energy.ge(self.energy_threshold),
            relative_energy.pow(self.energy_power),
            torch.zeros_like(relative_energy)).detach()
        student = F.normalize(student, dim=1, eps=self.norm_eps)
        teacher = F.normalize(teacher, dim=1, eps=self.norm_eps)
        return student, teacher, energy_weight

    def forward(self, student, teacher, teacher_heatmap,
                batch_data_samples=None, student_heatmap=None):
        if not self.enabled:
            return student.new_zeros(())
        student, teacher, energy_weight = self._prepare_features(
            student, teacher)
        loss = F.smooth_l1_loss(student, teacher, reduction='none')
        weight = self.spatial_weight(teacher_heatmap, loss.dtype)
        weight = weight * energy_weight
        return self.loss_weight * (loss * weight).sum() / (
            weight.sum() * loss.shape[1]).clamp_min(1)


@MODELS.register_module()
class ClassAwareBEVFeatureDistillLoss(BEVFeatureDistillLoss):
    """BEV feature KD with per-GT-class spatial reliability weights.

    Class weights affect only feature distillation inside each augmented GT
    box.  The student's ordinary detection losses remain unchanged.
    """

    def __init__(self, point_cloud_range, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.pc_range = tuple(float(value) for value in point_cloud_range)
        self.class_weights = tuple(float(value) for value in class_weights)
        if not self.class_weights:
            raise ValueError('class_weights must not be empty')
        if any(weight < 0 for weight in self.class_weights):
            raise ValueError('class_weights must be non-negative')

    def _class_weight_map(self, feature, batch_data_samples):
        batch, _, height, width = feature.shape
        weight_map = feature.new_ones((batch, 1, height, width),
                                      dtype=torch.float32)
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        grid_x = torch.linspace(
            x_min + (x_max - x_min) / (2 * height),
            x_max - (x_max - x_min) / (2 * height),
            height, device=feature.device, dtype=torch.float32)
        grid_y = torch.linspace(
            y_min + (y_max - y_min) / (2 * width),
            y_max - (y_max - y_min) / (2 * width),
            width, device=feature.device, dtype=torch.float32)
        # This project keeps BEV tensor axes as [x, y], consistent with the
        # transposed center used by TransFusionHead.get_targets.
        world_x, world_y = torch.meshgrid(grid_x, grid_y, indexing='ij')

        for batch_index, sample in enumerate(batch_data_samples):
            boxes = sample.gt_instances_3d.bboxes_3d.tensor.to(
                feature.device).float()
            labels = sample.gt_instances_3d.labels_3d.to(feature.device)
            for box, label in zip(boxes, labels):
                label_index = int(label.item())
                if label_index >= len(self.class_weights):
                    raise ValueError(
                        f'Class label {label_index} is outside class_weights '
                        f'of length {len(self.class_weights)}')
                class_weight = self.class_weights[label_index]
                if class_weight >= 1:
                    continue
                delta_x = world_x - box[0]
                delta_y = world_y - box[1]
                cos_yaw, sin_yaw = box[6].cos(), box[6].sin()
                local_x = delta_x * cos_yaw + delta_y * sin_yaw
                local_y = -delta_x * sin_yaw + delta_y * cos_yaw
                inside = ((local_x.abs() <= box[3] * 0.5) &
                          (local_y.abs() <= box[4] * 0.5))
                region_weight = torch.where(
                    inside, weight_map.new_tensor(class_weight),
                    weight_map.new_tensor(1.0))
                weight_map[batch_index, 0] = torch.minimum(
                    weight_map[batch_index, 0], region_weight)
        return weight_map.detach()

    def forward(self, student, teacher, teacher_heatmap,
                batch_data_samples=None, student_heatmap=None):
        if not self.enabled:
            return student.new_zeros(())
        if batch_data_samples is None:
            raise ValueError('Class-aware feature KD requires data samples')
        student, teacher, energy_weight = self._prepare_features(
            student, teacher)
        loss = F.smooth_l1_loss(student, teacher, reduction='none')
        weight = self.spatial_weight(teacher_heatmap, loss.dtype)
        weight = weight * energy_weight
        weight = weight * self._class_weight_map(
            teacher, batch_data_samples)
        return self.loss_weight * (loss * weight).sum() / (
            weight.sum() * loss.shape[1]).clamp_min(1)


@MODELS.register_module()
class TeacherReliableBEVFeatureDistillLoss(BEVFeatureDistillLoss):
    """Distill only where the teacher is correct and better than the student."""

    def __init__(self,
                 point_cloud_range,
                 min_teacher_confidence=0.001,
                 box_expand=1.5,
                 require_teacher_better=True,
                 class_weights=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pc_range = tuple(float(value) for value in point_cloud_range)
        self.min_teacher_confidence = float(min_teacher_confidence)
        self.box_expand = float(box_expand)
        self.require_teacher_better = bool(require_teacher_better)
        self.class_weights = (None if class_weights is None else
                              tuple(float(value) for value in class_weights))
        if not 0 <= self.min_teacher_confidence <= 1:
            raise ValueError('min_teacher_confidence must be in [0, 1]')
        if self.box_expand <= 0:
            raise ValueError('box_expand must be positive')
        if (self.class_weights is not None and
                any(weight < 0 for weight in self.class_weights)):
            raise ValueError('class_weights must be non-negative')

    def _reliability_map(self, feature, teacher_heatmap, student_heatmap,
                         batch_data_samples):
        batch, _, height, width = feature.shape
        reliability = feature.new_zeros((batch, 1, height, width),
                                        dtype=torch.float32)
        teacher_prob = teacher_heatmap.detach().float().sigmoid()
        student_prob = student_heatmap.detach().float().sigmoid()
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        grid_x = torch.linspace(
            x_min + (x_max - x_min) / (2 * height),
            x_max - (x_max - x_min) / (2 * height),
            height, device=feature.device, dtype=torch.float32)
        grid_y = torch.linspace(
            y_min + (y_max - y_min) / (2 * width),
            y_max - (y_max - y_min) / (2 * width),
            width, device=feature.device, dtype=torch.float32)
        world_x, world_y = torch.meshgrid(grid_x, grid_y, indexing='ij')

        for batch_index, sample in enumerate(batch_data_samples):
            boxes = sample.gt_instances_3d.bboxes_3d.tensor.to(
                feature.device).float()
            labels = sample.gt_instances_3d.labels_3d.to(feature.device)
            for box, label in zip(boxes, labels):
                center_x = ((box[0] - x_min) / (x_max - x_min) *
                            height).to(torch.long)
                center_y = ((box[1] - y_min) / (y_max - y_min) *
                            width).to(torch.long)
                if not (0 <= center_x < height and 0 <= center_y < width):
                    continue
                teacher_scores = teacher_prob[
                    batch_index, :, center_x, center_y]
                label_index = int(label.item())
                class_weight = 1.0
                if self.class_weights is not None:
                    if label_index >= len(self.class_weights):
                        raise ValueError(
                            f'Class label {label_index} is outside '
                            f'class_weights of length {len(self.class_weights)}')
                    class_weight = self.class_weights[label_index]
                    if class_weight <= 0:
                        continue
                teacher_score = teacher_scores[label_index]
                student_score = student_prob[
                    batch_index, label_index, center_x, center_y]
                if teacher_scores.argmax() != label:
                    continue
                if teacher_score < self.min_teacher_confidence:
                    continue
                if (self.require_teacher_better and
                        teacher_score <= student_score):
                    continue

                delta_x = world_x - box[0]
                delta_y = world_y - box[1]
                cos_yaw, sin_yaw = box[6].cos(), box[6].sin()
                local_x = delta_x * cos_yaw + delta_y * sin_yaw
                local_y = -delta_x * sin_yaw + delta_y * cos_yaw
                inside = (
                    (local_x.abs() <= box[3] * self.box_expand * 0.5) &
                    (local_y.abs() <= box[4] * self.box_expand * 0.5))
                region = torch.where(
                    inside, teacher_score * class_weight,
                    teacher_score.new_zeros(()))
                reliability[batch_index, 0] = torch.maximum(
                    reliability[batch_index, 0], region)
        return reliability.detach()

    def forward(self, student, teacher, teacher_heatmap,
                batch_data_samples=None, student_heatmap=None):
        if not self.enabled:
            return student.new_zeros(())
        if batch_data_samples is None or student_heatmap is None:
            raise ValueError(
                'Teacher-reliable KD requires samples and student heatmap')
        student, teacher, energy_weight = self._prepare_features(
            student, teacher)
        loss = F.smooth_l1_loss(student, teacher, reduction='none')
        weight = self._reliability_map(
            teacher, teacher_heatmap, student_heatmap, batch_data_samples)
        weight = weight * energy_weight
        return self.loss_weight * (loss * weight).sum() / (
            weight.sum() * loss.shape[1]).clamp_min(1)


@MODELS.register_module()
class HeatmapDistillLoss(_WeightedDistillLoss):
    def __init__(self, temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = float(temperature)

    def forward(self, student, teacher):
        if not self.enabled:
            return student.new_zeros(())
        teacher = teacher.detach().float()
        student = student.float()
        target = (teacher / self.temperature).sigmoid()
        loss = F.binary_cross_entropy_with_logits(
            student / self.temperature, target, reduction='none')
        weight = self.spatial_weight(
            teacher, loss.dtype, temperature=self.temperature)
        return (self.loss_weight * self.temperature**2 *
                (loss * weight).sum() /
                (weight.sum() * loss.shape[1]).clamp_min(1))


@MODELS.register_module()
class BEVAttentionDistillLoss(_WeightedDistillLoss):
    def forward(self, student, teacher, teacher_heatmap):
        if not self.enabled:
            return student.new_zeros(())
        student_map = student.float().abs().mean(dim=1, keepdim=True)
        teacher_map = teacher.detach().float().abs().mean(dim=1, keepdim=True)
        student_map = F.normalize(student_map.flatten(1), dim=1).view_as(student_map)
        teacher_map = F.normalize(teacher_map.flatten(1), dim=1).view_as(teacher_map)
        loss = F.mse_loss(student_map, teacher_map, reduction='none')
        weight = self.spatial_weight(teacher_heatmap, loss.dtype)
        return self.loss_weight * (loss * weight).sum() / weight.sum().clamp_min(1)


class _InstanceDistillLoss(nn.Module):
    """Utilities for sparse object-centric BEV distillation."""

    def __init__(self, point_cloud_range, loss_weight=1.0, enabled=True,
                 norm_eps=1e-4, energy_threshold=0.01):
        super().__init__()
        self.pc_range = tuple(float(value) for value in point_cloud_range)
        self.loss_weight = float(loss_weight)
        self.enabled = bool(enabled)
        self.norm_eps = float(norm_eps)
        self.energy_threshold = float(energy_threshold)
        if not 0 <= self.energy_threshold < 1:
            raise ValueError('energy_threshold must be in [0, 1)')

    @staticmethod
    def _nine_points(boxes):
        """Return center, four corners and four edge midpoints in BEV."""
        centers = boxes[:, :2]
        half_dims = boxes[:, 3:5] * 0.5
        pattern = boxes.new_tensor([
            [0., 0.], [-1., -1.], [-1., 1.], [1., 1.], [1., -1.],
            [-1., 0.], [0., 1.], [1., 0.], [0., -1.]
        ])
        offsets = pattern[None] * half_dims[:, None]
        yaw = boxes[:, 6]
        cos_yaw, sin_yaw = yaw.cos(), yaw.sin()
        rotation = torch.stack([
            cos_yaw, -sin_yaw, sin_yaw, cos_yaw
        ], dim=-1).reshape(-1, 2, 2)
        offsets = torch.matmul(offsets, rotation.transpose(1, 2))
        return centers[:, None] + offsets

    def _sample(self, feature, boxes):
        boxes = boxes.to(device=feature.device)
        if boxes.numel() == 0:
            return feature.new_empty((0, 9, feature.shape[1]),
                                     dtype=torch.float32)
        points = self._nine_points(boxes.float())
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        grid_x = 2 * (points[..., 0] - x_min) / (x_max - x_min) - 1
        grid_y = 2 * (points[..., 1] - y_min) / (y_max - y_min) - 1
        # grid_sample expects [last-axis, first-axis].  The project BEV layout
        # is [x, y], therefore its sampling grid is [y, x].
        grid = torch.stack([grid_y, grid_x], dim=-1).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            feature.float(), grid, mode='bilinear', padding_mode='zeros',
            align_corners=False)
        return sampled.squeeze(0).squeeze(-1).transpose(0, 1).reshape(
            boxes.shape[0], 9, feature.shape[1])

    @staticmethod
    def _boxes(sample):
        return sample.gt_instances_3d.bboxes_3d.tensor


@MODELS.register_module()
class InstanceFeatureDistillLoss(_InstanceDistillLoss):
    """UniDistill-style feature alignment at nine points per GT box."""

    def forward(self, student, teacher, batch_data_samples):
        if not self.enabled:
            return student.new_zeros(())
        per_box = []
        for index, sample in enumerate(batch_data_samples):
            boxes = self._boxes(sample)
            if boxes.numel() == 0:
                continue
            student_points = self._sample(student[index:index + 1], boxes)
            teacher_points = self._sample(
                teacher[index:index + 1].detach(), boxes)
            per_box.append((student_points - teacher_points).abs().mean(
                dim=(-2, -1)))
        if not per_box:
            return student.new_zeros(())
        return self.loss_weight * torch.cat(per_box).mean()


@MODELS.register_module()
class InstanceRelationDistillLoss(_InstanceDistillLoss):
    """Align pairwise relations among each object's nine BEV points."""

    def forward(self, student, teacher, batch_data_samples):
        if not self.enabled:
            return student.new_zeros(())
        per_box = []
        for index, sample in enumerate(batch_data_samples):
            boxes = self._boxes(sample)
            if boxes.numel() == 0:
                continue
            student_points = self._sample(
                student[index:index + 1], boxes)
            teacher_points = self._sample(
                teacher[index:index + 1].detach(), boxes)
            teacher_energy = teacher_points.norm(dim=-1)
            relative_energy = teacher_energy / teacher_energy.amax(
                dim=-1, keepdim=True).clamp_min(self.norm_eps)
            valid_point = (
                teacher_energy.gt(self.norm_eps) &
                relative_energy.ge(self.energy_threshold))
            pair_weight = (
                valid_point.unsqueeze(-1) & valid_point.unsqueeze(-2)
            ).to(dtype=torch.float32)
            student_points = F.normalize(
                student_points, dim=-1, eps=self.norm_eps)
            teacher_points = F.normalize(
                teacher_points, dim=-1, eps=self.norm_eps)
            student_relation = torch.matmul(
                student_points, student_points.transpose(-1, -2))
            teacher_relation = torch.matmul(
                teacher_points, teacher_points.transpose(-1, -2))
            # Keep every GT box equally weighted when batch_size changes.
            # Skip boxes whose every sampled point fell on zero-padded BEV
            # boundary cells, otherwise they dilute the per-box average with
            # a zero contribution.
            pair_sum = pair_weight.sum(dim=(-2, -1))
            valid = pair_sum.gt(0)
            if not valid.any():
                continue
            difference = (student_relation[valid] -
                           teacher_relation[valid]).abs()
            per_box.append((difference * pair_weight[valid]).sum(
                dim=(-2, -1)) / pair_sum[valid])
        if not per_box:
            return student.new_zeros(())
        return self.loss_weight * torch.cat(per_box).mean()


@MODELS.register_module()
class GaussianResponseDistillLoss(_InstanceDistillLoss):
    """UniDistill response alignment inside GT Gaussian regions."""

    def __init__(self, gaussian_overlap=0.1, min_radius=2, **kwargs):
        super().__init__(**kwargs)
        self.gaussian_overlap = float(gaussian_overlap)
        self.min_radius = int(min_radius)

    def _foreground_mask(self, heatmap, batch_data_samples):
        batch, classes, height, width = heatmap.shape
        mask = heatmap.new_zeros((batch, 1, height, width),
                                 dtype=torch.float32)
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        for batch_index, sample in enumerate(batch_data_samples):
            boxes = self._boxes(sample).to(heatmap.device).float()
            for box in boxes:
                center_x = (box[0] - x_min) / (x_max - x_min) * height
                center_y = (box[1] - y_min) / (y_max - y_min) * width
                if not (0 <= center_x < height and 0 <= center_y < width):
                    continue
                box_width = box[3] / (x_max - x_min) * height
                box_length = box[4] / (y_max - y_min) * width
                radius = gaussian_radius(
                    (box_length, box_width),
                    min_overlap=self.gaussian_overlap)
                radius = max(self.min_radius, int(radius))
                # Match TransFusionHead.get_targets, whose project-specific
                # BEV layout passes [y, x] into draw_heatmap_gaussian.
                center = torch.stack([center_y, center_x])
                draw_heatmap_gaussian(
                    mask[batch_index, 0], center.to(torch.int32), radius)
        return mask

    def forward(self, student, teacher, batch_data_samples):
        if not self.enabled:
            return student.new_zeros(())
        # TransFusion exposes a dense classification response but query-based
        # regression, so use the classification-max variant validated by the
        # UniDistill ablation instead of inventing a dense regression map.
        student_response = student.float().sigmoid().amax(dim=1, keepdim=True)
        teacher_response = teacher.detach().float().sigmoid().amax(
            dim=1, keepdim=True)
        foreground = self._foreground_mask(teacher, batch_data_samples)
        num_boxes = sum(len(self._boxes(sample)) for sample in batch_data_samples)
        difference = (student_response - teacher_response).abs()
        return (self.loss_weight * (difference * foreground).sum() /
                max(num_boxes, 1))


@MODELS.register_module()
class DFBEVFusionLidarDistiller(Base3DDetector):
    """Frozen BEVFusion teacher with a deployable DFBEVFusion student."""

    def __init__(self,
                 teacher,
                 student,
                 teacher_checkpoint: Optional[str] = None,
                 student_checkpoint: Optional[str] = None,
                 bev_loss=None,
                 heatmap_loss=None,
                 attention_loss=None,
                 instance_feature_loss=None,
                 middle_feature_loss=None,
                 instance_relation_loss=None,
                 gaussian_response_loss=None,
                 warmup_epochs=2,
                 warmup_iters=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.teacher = MODELS.build(deepcopy(teacher))
        self.student = MODELS.build(deepcopy(student))
        self.bev_loss = MODELS.build(bev_loss)
        self.heatmap_loss = MODELS.build(heatmap_loss)
        self.attention_loss = MODELS.build(attention_loss)
        self.instance_feature_loss = MODELS.build(instance_feature_loss)
        self.middle_feature_loss = (MODELS.build(middle_feature_loss)
                                    if middle_feature_loss is not None else None)
        self.instance_relation_loss = MODELS.build(instance_relation_loss)
        self.gaussian_response_loss = MODELS.build(gaussian_response_loss)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_iters = (None if warmup_iters is None else
                             int(warmup_iters))
        if self.warmup_epochs < 0:
            raise ValueError('warmup_epochs must be non-negative')
        if self.warmup_iters is not None and self.warmup_iters < 0:
            raise ValueError('warmup_iters must be non-negative')

        if teacher_checkpoint:
            self._load_strict_checkpoint(self.teacher, teacher_checkpoint,
                                         'teacher')
        if student_checkpoint:
            self._load_strict_checkpoint(self.student, student_checkpoint,
                                         'student')
        self._freeze_teacher()

    @staticmethod
    def _load_strict_checkpoint(model, filename, role):
        # These are explicitly configured, trusted local training artifacts.
        # Explicit weights_only=False keeps compatibility with PyTorch 2.6 and
        # older MMEngine checkpoints containing HistoryBuffer metadata.
        checkpoint = torch.load(filename, map_location='cpu',
                                weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing = ('pts_middle_adapter.', ) if role == 'student' else ()
        illegal_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing)
        ]
        if illegal_missing or incompatible.unexpected_keys:
            details = (f'missing keys={illegal_missing}, unexpected keys='
                       f'{incompatible.unexpected_keys}')
            raise RuntimeError(
                f'Failed to strictly load {role} checkpoint {filename!r}: '
                f'{details}')

    def _freeze_teacher(self):
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _loss_enabled(loss_module):
        return (
            loss_module is not None
            and getattr(loss_module, 'enabled', True)
            and float(getattr(loss_module, 'loss_weight', 1.0)) != 0.0)

    @staticmethod
    def _dense_heatmap(detector, bev_feat):
        """Run only shared-conv + dense heatmap branch, not TransFusion."""
        if not isinstance(bev_feat, torch.Tensor):
            if len(bev_feat) != 1:
                raise ValueError(
                    f'Dense heatmap requires one BEV level, got {len(bev_feat)}')
            bev_feat = bev_feat[0]
        fusion_feat = detector.bbox_head.shared_conv(bev_feat)
        with torch.autocast('cuda', enabled=False):
            return detector.bbox_head.heatmap_head(fusion_feat.float())

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _warmup_factor(self):
        hub = MessageHub.get_current_instance()
        iteration = hub.get_info('iter')
        iteration = 0 if iteration is None else int(iteration)
        if self.warmup_iters is not None:
            if self.warmup_iters == 0:
                return 1.0
            return min(1.0, max(0.0,
                                (iteration + 1) / self.warmup_iters))
        if self.warmup_epochs == 0:
            return 1.0

        # MMEngine publishes max_iters/max_epochs during normal Runner
        # training.  Use them to include fractional epoch progress instead of
        # applying one constant KD factor for an entire epoch.
        max_iters = hub.get_info('max_iters')
        max_epochs = hub.get_info('max_epochs')
        if max_iters and max_epochs:
            epoch_progress = ((iteration + 1) * float(max_epochs) /
                              float(max_iters))
            return min(1.0, max(0.0,
                                epoch_progress / self.warmup_epochs))

        # Conservative fallback for non-Runner unit tests or custom loops.
        epoch = hub.get_info('epoch')
        if epoch is None:
            epoch = 0
        return min(1.0, max(0.0, (int(epoch) + 1) / self.warmup_epochs))

    @staticmethod
    def _validate_outputs(student, teacher):
        student_low = student['low_bev_feat']
        teacher_low = teacher['low_bev_feat']
        if student_low.shape != teacher_low.shape:
            raise ValueError('Low-level BEV distillation shape mismatch: '
                             f'student {tuple(student_low.shape)} vs teacher '
                             f'{tuple(teacher_low.shape)}')
        student_middle = student['middle_bev_feat']
        teacher_middle = teacher['middle_bev_feat']
        if student_middle.shape != teacher_middle.shape:
            raise ValueError('Middle BEV distillation shape mismatch: '
                             f'student {tuple(student_middle.shape)} vs teacher '
                             f'{tuple(teacher_middle.shape)}')
        student_bev, teacher_bev = student['bev_feat'], teacher['bev_feat']
        if student_bev.shape != teacher_bev.shape:
            raise ValueError('BEV distillation shape mismatch: student '
                             f'{tuple(student_bev.shape)} vs teacher '
                             f'{tuple(teacher_bev.shape)}')

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample], **kwargs):
        metas = [sample.metainfo for sample in batch_data_samples]
        student_outputs = self.student.extract_distill_features(
            batch_inputs_dict, metas)

        # Run the full student TransFusionHead exactly once, then reuse its
        # predictions for both supervised loss and optional heatmap KD.
        student_preds = self.student.bbox_head(
            student_outputs['bev_feat'], metas)
        batch_gt_instances_3d = [
            sample.gt_instances_3d for sample in batch_data_samples
        ]
        losses = self.student.bbox_head.loss_by_feat(
            student_preds, batch_gt_instances_3d, **kwargs)
        student_heatmap = student_preds[0][0]['dense_heatmap']

        modules = (
            self.bev_loss, self.heatmap_loss, self.attention_loss,
            self.instance_feature_loss, self.middle_feature_loss,
            self.instance_relation_loss, self.gaussian_response_loss)
        active = [self._loss_enabled(module) for module in modules]
        need_teacher = any(active)
        need_heatmap = any((active[0], active[1], active[2],
                            active[4], active[6]))

        teacher_outputs = None
        teacher_heatmap = None
        if need_teacher:
            with torch.no_grad():
                teacher_outputs = self.teacher.extract_distill_features(
                    batch_inputs_dict, metas)
                teacher_outputs = {
                    key: value.detach()
                    for key, value in teacher_outputs.items()
                }
                if need_heatmap:
                    teacher_heatmap = self._dense_heatmap(
                        self.teacher, teacher_outputs['bev_feat']).detach()
            self._validate_outputs(student_outputs, teacher_outputs)
            if (need_heatmap and
                    student_heatmap.shape != teacher_heatmap.shape):
                raise ValueError(
                    'Heatmap/class mismatch: student '
                    f'{tuple(student_heatmap.shape)} vs teacher '
                    f'{tuple(teacher_heatmap.shape)}')

        factor = self._warmup_factor()
        zero = student_outputs['bev_feat'].new_zeros(())
        losses['loss_kd_bev'] = zero
        losses['loss_kd_heatmap'] = zero
        losses['loss_kd_attention'] = zero
        losses['loss_kd_instance_feature'] = zero
        losses['loss_kd_instance_relation'] = zero
        losses['loss_kd_gaussian_response'] = zero
        if self.middle_feature_loss is not None:
            losses['loss_kd_middle_feature'] = zero

        if active[0]:
            losses['loss_kd_bev'] = factor * self.bev_loss(
                student_outputs['bev_feat'], teacher_outputs['bev_feat'],
                teacher_heatmap)
        if active[1]:
            losses['loss_kd_heatmap'] = factor * self.heatmap_loss(
                student_heatmap, teacher_heatmap)
        if active[2]:
            losses['loss_kd_attention'] = factor * self.attention_loss(
                student_outputs['bev_feat'], teacher_outputs['bev_feat'],
                teacher_heatmap)
        if active[3]:
            losses['loss_kd_instance_feature'] = factor * (
                self.instance_feature_loss(
                    student_outputs['low_bev_feat'],
                    teacher_outputs['low_bev_feat'],
                    batch_data_samples))
        if active[4]:
            losses['loss_kd_middle_feature'] = factor * (
                self.middle_feature_loss(
                    student_outputs['middle_bev_feat'],
                    teacher_outputs['middle_bev_feat'],
                    teacher_heatmap, batch_data_samples,
                    student_heatmap))
        if active[5]:
            losses['loss_kd_instance_relation'] = factor * (
                self.instance_relation_loss(
                    student_outputs['bev_feat'], teacher_outputs['bev_feat'],
                    batch_data_samples))
        if active[6]:
            losses['loss_kd_gaussian_response'] = factor * (
                self.gaussian_response_loss(
                    student_heatmap, teacher_heatmap, batch_data_samples))
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        return self.student.predict(batch_inputs_dict, batch_data_samples,
                                    **kwargs)

    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        return self.student.extract_feat(batch_inputs_dict, batch_input_metas,
                                         **kwargs)

    def _forward(self, batch_inputs, batch_data_samples=None, **kwargs):
        return self.student._forward(batch_inputs, batch_data_samples, **kwargs)
