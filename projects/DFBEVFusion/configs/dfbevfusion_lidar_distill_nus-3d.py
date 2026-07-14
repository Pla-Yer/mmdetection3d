"""Full nuScenes LiDAR distillation template.

Fill or override both checkpoint variables before training. No checkpoint is
downloaded or converted by this configuration.
"""

from copy import deepcopy

from mmengine.config import Config

_base_ = ['./bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

custom_imports = dict(
    imports=['projects.DFBEVFusion.bevfusion'],
    allow_failed_imports=False)

teacher_checkpoint = None  # Full BEVFusion LiDAR checkpoint path.
student_checkpoint = None  # Optional DFBEVFusion LiDAR initialization path.

teacher_cfg = Config.fromfile(
    'projects/BEVFusion/configs/'
    'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
    import_custom_modules=False)
teacher_model = deepcopy(teacher_cfg.model)
teacher_model['type'] = 'DistillBEVFusionTeacher'
student_model = deepcopy(_base_.model)
distill_data_preprocessor = deepcopy(student_model['data_preprocessor'])
distill_data_preprocessor.pop('voxelize_cfg')

model = dict(
    _delete_=True,
    type='DFBEVFusionLidarDistiller',
    data_preprocessor=distill_data_preprocessor,
    teacher=teacher_model,
    student=student_model,
    teacher_checkpoint=teacher_checkpoint,
    student_checkpoint=student_checkpoint,
    warmup_epochs=2,
    bev_loss=dict(
        type='BEVFeatureDistillLoss', loss_weight=0.0, enabled=False,
        norm_eps=1e-4, confidence_threshold=0.1, background_weight=0.1),
    heatmap_loss=dict(
        type='HeatmapDistillLoss', loss_weight=0.0, enabled=False,
        temperature=2.0, confidence_threshold=0.1,
        background_weight=0.1),
    attention_loss=dict(
        type='BEVAttentionDistillLoss', loss_weight=0.0, enabled=False,
        confidence_threshold=0.1, background_weight=0.1),
    instance_feature_loss=dict(
        type='InstanceFeatureDistillLoss',
        point_cloud_range=[-54., -54., -5., 54., 54., 3.],
        loss_weight=10.0, enabled=False, norm_eps=1e-4),
    instance_relation_loss=dict(
        type='InstanceRelationDistillLoss',
        point_cloud_range=[-54., -54., -5., 54., 54., 3.],
        loss_weight=1.0, norm_eps=1e-4),
    gaussian_response_loss=dict(
        type='GaussianResponseDistillLoss',
        point_cloud_range=[-54., -54., -5., 54., 54., 3.],
        loss_weight=10.0, enabled=False,
        gaussian_overlap=0.1, min_radius=2))

train_dataloader = dict(batch_size=1)

# A conservative initial scale avoids intermittent fp16 overflow through
# sparse object-point sampling while retaining dynamic down-scaling.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=dict(init_scale=512.0, growth_interval=2000))

del Config, deepcopy, teacher_cfg, teacher_model, student_model
del distill_data_preprocessor
