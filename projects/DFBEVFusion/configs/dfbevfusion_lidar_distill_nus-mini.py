"""nuScenes-mini LiDAR distillation configuration.

Checkpoint paths are deliberately top-level variables so they can be
overridden with ``--cfg-options`` without editing model implementation code.
"""

from copy import deepcopy

from mmengine.config import Config

_base_ = ['./bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-6e_nus-3d.py']

custom_imports = dict(
    imports=['projects.DFBEVFusion.bevfusion'],
    allow_failed_imports=False)

teacher_checkpoint = 'work_dirs/bevfusion_nus_mini_lidar/epoch_15.pth'
student_checkpoint = 'work_dirs/dfbevfusion_nus_mini_lidar/epoch_15.pth'

teacher_cfg = Config.fromfile(
    'projects/BEVFusion/configs/'
    'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-6e_nus-mini-3d.py',
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
        type='BEVFeatureDistillLoss',
        loss_weight=0.0,
        enabled=False,
        norm_eps=1e-4,
        confidence_threshold=0.1,
        background_weight=0.1),
    heatmap_loss=dict(
        type='HeatmapDistillLoss',
        loss_weight=0.0,
        enabled=False,
        temperature=2.0,
        confidence_threshold=0.1,
        background_weight=0.1),
    attention_loss=dict(
        type='BEVAttentionDistillLoss',
        loss_weight=0.0,
        enabled=False,
        confidence_threshold=0.1,
        background_weight=0.1),
    instance_feature_loss=dict(
        type='InstanceFeatureDistillLoss',
        point_cloud_range=[-54., -54., -5., 54., 54., 3.],
        loss_weight=10.0,
        enabled=False,
        norm_eps=1e-4),
    instance_relation_loss=dict(
        type='InstanceRelationDistillLoss',
        point_cloud_range=[-54., -54., -5., 54., 54., 3.],
        loss_weight=1.0,
        norm_eps=1e-4),
    gaussian_response_loss=dict(
        type='GaussianResponseDistillLoss',
        point_cloud_range=[-54., -54., -5., 54., 54., 3.],
        loss_weight=10.0,
        enabled=False,
        gaussian_overlap=0.1,
        min_radius=2))

train_dataloader = dict(batch_size=1)

# The student is fine-tuned from epoch 15 rather than trained from scratch.
# Keep this below the original 1e-4 base learning rate to avoid destroying the
# pretrained detector and to reduce AMP gradient overflows.
lr = 1e-5
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(lr=lr),
    loss_scale=dict(init_scale=512.0, growth_interval=2000))
param_scheduler = [
    dict(
        type='CosineAnnealingLR', begin=0, T_max=3, end=3,
        by_epoch=True, eta_min_ratio=1e-2,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum', eta_min=0.85 / 0.95,
        begin=0, end=1.2, by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum', eta_min=1,
        begin=1.2, end=3, by_epoch=True,
        convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=3, val_interval=1)

del Config, deepcopy, teacher_cfg, teacher_model, student_model
del distill_data_preprocessor
