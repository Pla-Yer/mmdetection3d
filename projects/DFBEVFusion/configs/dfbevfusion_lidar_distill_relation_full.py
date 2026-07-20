"""Full-nuScenes LiDAR relation distillation for DFBEVFusion.

This configuration fine-tunes a trained DFBEVFusion LiDAR student under a
frozen original sparse-voxel BEVFusion LiDAR teacher.  The only enabled KD
objective is object-centric relation alignment on the final 512x180x180 BEV
feature.  All raw-feature and response losses remain disabled.

The base below must be the exact full-nuScenes student configuration used to
train the checkpoint supplied through ``student_checkpoint``.
"""

from copy import deepcopy

from mmengine.config import Config


_base_ = [
    './dfbevfusion_lidar_voxel03_second_secfpn_1x4-amp-acc8-cyclic-20e_nus-full-5090.py'
]

custom_imports = dict(
    imports=['projects.DFBEVFusion.bevfusion'],
    allow_failed_imports=False)

# Defaults are placeholders. Prefer overriding both paths from the command
# line so the experiment configuration remains reusable.
teacher_checkpoint = '/root/autodl-tmp/mmdetection3d/teacher_model/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth'
student_checkpoint = 'work_dirs/dfbevfusion_lidar_full_5090/best_NuScenes_NDS_epoch_15.pth'

teacher_cfg = Config.fromfile(
    'projects/BEVFusion/configs/'
    'bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
    import_custom_modules=False)
teacher_model = deepcopy(teacher_cfg.model)
teacher_model['type'] = 'DistillBEVFusionTeacher'

student_model = deepcopy(_base_.model)

# The wrapper performs common point/data preprocessing.  The teacher and
# student retain their own voxel layers, so the wrapper must not voxelize a
# third time.
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

    # Rejected/unused KD objectives are kept explicitly disabled so the
    # relation-only experiment differs from its matched control in one flag.
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
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        loss_weight=0.0,
        enabled=False,
        norm_eps=1e-4),
    instance_relation_loss=dict(
        type='InstanceRelationDistillLoss',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        loss_weight=1.0,
        enabled=True,
        norm_eps=1e-4),
    gaussian_response_loss=dict(
        type='GaussianResponseDistillLoss',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        loss_weight=0.0,
        enabled=False,
        gaussian_overlap=0.1,
        min_radius=2))

# A frozen teacher and a trainable student coexist on the GPU.  Start with one
# sample per GPU; increase only after measuring real peak memory.
train_dataloader = dict(batch_size=4,num_workers=4)

randomness = dict(seed=20260719, deterministic=False)

lr = 1e-5
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    loss_scale=dict(init_scale=512.0, growth_interval=2000),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=3,
        T_max=3,
        by_epoch=True,
        eta_min_ratio=1e-2,
        convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=3, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=3))

# The base student may disable database sampling near epoch 15.  That hook is
# irrelevant to a three-epoch fine-tune and would introduce an unnecessary
# inherited variable.
custom_hooks = []

del Config, deepcopy, teacher_cfg, teacher_model, student_model
del distill_data_preprocessor

