"""DFBEVFusion LiDAR-camera training on full nuScenes with ResNet50.

This config directly inherits the validated full-nuScenes LiDAR-only
DFBEVFusion config. It keeps the complete 0.30 m PointPillars LiDAR branch
and adds the ResNet50 camera branch used by NVIDIA CUDA-BEVFusion:

    ResNet50 -> GeneralizedLSSFPN(256) -> DepthLSSTransform(80)
             -> CUDA BEVPool -> 2x BEV downsample
    PointPillars -> BEVDownsample(256)
    ConvFuser([80, 256] -> 256) -> SECOND -> SECONDFPN -> TransFusionHead

Spatial alignment:
    camera BEV: 80 x 360 x 360 -> downsample=2 -> 80 x 180 x 180
    LiDAR BEV:  256 x 360 x 360 -> BEVDownsample -> 256 x 180 x 180

The local implementation of ``DepthLSSTransform`` is expected to use the
restored CUDA BEVPool operator.
"""

_base_ = [
    './dfbevfusion_lidar_voxel03_second_secfpn_1x4-amp-acc8-cyclic-20e_nus-full-5090.py'
]

# -------------------------------------------------------------------------
# Dataset and geometry
# -------------------------------------------------------------------------

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.30, 0.30, 8.0]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

metainfo = dict(classes=class_names, version='v1.0-trainval')
data_root = 'data/nuscenes/'
backend_args = None

input_modality = dict(use_lidar=True, use_camera=True)

# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------

model = dict(
    type='DFBEVFusion',

    # Keep the inherited LiDAR voxelization settings explicit and add
    # ImageNet normalization for the RGB camera input.
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        voxelize_cfg=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[30000, 40000],
            voxelize_reduce=False)),

    # NVIDIA CUDA-BEVFusion ResNet50 camera backbone.
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                'https://download.pytorch.org/models/'
                'resnet50-0676ba61.pth'))),

    # BEVFusion generalized LSS FPN. Its highest-resolution output is
    # 256 x 32 x 88 for a 256 x 704 image.
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),

    # The 0.30 m camera BEV is initially 360 x 360. downsample=2 aligns it
    # with the inherited LiDAR BEVDownsample output at 180 x 180.
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.30],
        ybound=[-54.0, 54.0, 0.30],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2),

    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[80, 256],
        out_channels=256))

# -------------------------------------------------------------------------
# Data pipelines
# -------------------------------------------------------------------------

# ObjectSample is intentionally not used: the current database sampler inserts
# LiDAR points and 3D boxes but does not paste corresponding objects into the
# six camera images, which would break cross-modal consistency.
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
            'cam2lidar', 'ori_lidar2img', 'img_aug_matrix',
            'lidar_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow',
            'pcd_rotation', 'pcd_scale_factor', 'pcd_trans',
            'num_pts_feats', 'num_views'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
            'cam2lidar', 'ori_lidar2img', 'img_aug_matrix',
            'lidar_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'num_views'
        ])
]

# -------------------------------------------------------------------------
# Dataloaders and evaluation
# -------------------------------------------------------------------------

# One RTX 5090: micro-batch 2 with 16-step accumulation reproduces the
# nominal global batch size 32 used by the reference BEVFusion recipe.
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    dataset=dict(
        dataset=dict(
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality)))

val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    drop_last=False,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# -------------------------------------------------------------------------
# Optimizer and 6-epoch NVIDIA/BEVFusion schedule
# -------------------------------------------------------------------------

optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=16,
    optimizer=dict(type='AdamW', lr=2.0e-4, weight_decay=0.01),
    loss_scale=dict(init_scale=512.0, growth_interval=2000),
    clip_grad=dict(max_norm=35, norm_type=2))

# The original recipe warms up for 500 optimizer updates. With 16 micro-batches
# accumulated per update, use 500 * 16 dataloader iterations.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=8000),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=6,
        T_max=6,
        by_epoch=True,
        eta_min_ratio=1.0e-3,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(enable=False, base_batch_size=32)

# The inherited LiDAR-only ObjectSample hook is not applicable here.
custom_hooks = []

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/NDS',
        rule='greater'))

env_cfg = dict(cudnn_benchmark=True)
randomness = dict(seed=20260728, deterministic=False)

# -------------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------------

# Load the strongest matched no-KD LiDAR continuation checkpoint.
# Camera backbone is initialized separately by img_backbone.init_cfg;
# img_neck, view_transform, and fusion_layer start from random initialization.
load_from = (
    'work_dirs/dfbevfusion_lidar_full_nokd_control_ext/'
    'best_NuScenes_NDS_epoch_4_student_only.pth'
)
resume = False
