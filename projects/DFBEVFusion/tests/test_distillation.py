import torch
from mmengine.structures import InstanceData

from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from projects.DFBEVFusion.bevfusion.distillation import (
    ClassAwareBEVFeatureDistillLoss,
    TeacherReliableBEVFeatureDistillLoss)


def _sample(label):
    sample = Det3DDataSample()
    instances = InstanceData()
    instances.bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor([[0., 0., 0., 2., 2., 1., 0., 0., 0.]]), box_dim=9)
    instances.labels_3d = torch.tensor([label], dtype=torch.long)
    sample.gt_instances_3d = instances
    return sample


def _loss():
    return ClassAwareBEVFeatureDistillLoss(
        point_cloud_range=[-4., -4., -2., 4., 4., 2.],
        class_weights=[1., 1., 1., 0.],
        loss_weight=1.)


def test_bus_box_has_zero_feature_kd_weight():
    feature = torch.zeros(1, 2, 8, 8)
    weight = _loss()._class_weight_map(feature, [_sample(3)])

    assert weight[0, 0, 3:5, 3:5].eq(0).all()
    assert weight[0, 0, 0, 0] == 1


def test_non_bus_box_keeps_feature_kd_weight():
    feature = torch.zeros(1, 2, 8, 8)
    weight = _loss()._class_weight_map(feature, [_sample(0)])

    assert weight.eq(1).all()


def _reliable_loss():
    return TeacherReliableBEVFeatureDistillLoss(
        point_cloud_range=[-4., -4., -2., 4., 4., 2.],
        loss_weight=1., box_expand=1.)


def test_reliable_map_keeps_teacher_correct_and_better_box():
    feature = torch.zeros(1, 2, 8, 8)
    teacher_heatmap = torch.full((1, 4, 8, 8), -10.)
    student_heatmap = torch.full_like(teacher_heatmap, -10.)
    teacher_heatmap[0, 3, 4, 4] = 4.
    student_heatmap[0, 3, 4, 4] = 1.

    weight = _reliable_loss()._reliability_map(
        feature, teacher_heatmap, student_heatmap, [_sample(3)])

    assert weight[0, 0, 3:5, 3:5].gt(0).all()
    assert weight[0, 0, 0, 0] == 0


def test_reliable_map_rejects_wrong_or_weaker_teacher():
    feature = torch.zeros(1, 2, 8, 8)
    teacher_heatmap = torch.full((1, 4, 8, 8), -10.)
    student_heatmap = torch.full_like(teacher_heatmap, -10.)
    teacher_heatmap[0, 2, 4, 4] = 4.  # Wrong teacher class.
    teacher_heatmap[0, 3, 4, 4] = 2.
    student_heatmap[0, 3, 4, 4] = 3.  # Student is also stronger.

    weight = _reliable_loss()._reliability_map(
        feature, teacher_heatmap, student_heatmap, [_sample(3)])

    assert weight.eq(0).all()
