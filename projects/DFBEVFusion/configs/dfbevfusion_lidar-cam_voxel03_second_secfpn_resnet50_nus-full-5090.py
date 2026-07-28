"""DFBEVFusion LiDAR-camera training on full nuScenes with ResNet50 (5090).

Based on ``dfbevfusion_lidar_voxel03_second_secfpn_1x4-amp-acc8-cyclic-
20e_nus-full-5090.py`` (the validated 5090 LiDAR-only config) with camera
components added from ``bevfusion_lidar-cam_voxel03_second_secfpn_8xb4-
cyclic-20e_nus-3d_resnet50.py``.

The LiDAR branch keeps the exact same architecture as the 5090 LiDAR-
only model: PointPillarsScatter (256x360x360) -> BEVDownsample
(256x180x180) -> SECOND -> FPN -> TransFusionHead. The LiDAR branch is
initialized from the ep15 LiDAR-only checkpoint via ``load_from``.

The camera branch (ResNet50 + GeneralizedLSSFPN + DepthLSSTransform)
is added on top. ``view_transform.downsample=2`` halves the camera BEV
from 360x360 to 180x180, matching the LiDAR BEV after BEVDownsample so
ConvFuser can fuse them:

    camera: 80x180x180  (DepthLSSTransform + downsample=2)
    lidar:  256x180x180 (PointPillarsScatter + BEVDownsample)
    fused:  256x180x180 (ConvFuser in_channels=[80, 256])

5090 adaptations inherited from the base 5090 config:
- AMP (init_scale=512, growth_interval=2000)
- full nuScenes (version='v1.0-trainval', 30k/40k voxels)
- db_sampler with full nuScenes DB infos
- env_cfg, randomness, etc.

Changed from the base 5090 config for LiDAR+cam:
- batch_size 4 -> 2 (6 cameras per sample increase memory ~3x)
- accumulative_counts 8 -> 4 (effective batch 8)
- lr 1e-4 -> 2e-4 (from reference LiDAR+cam config)
- max_epochs 20 -> 6 (LiDAR+cam uses shorter schedule)
- param_scheduler: cyclic-20e -> 6-epoch cosine + momentum (from reference)
- val_interval 5 -> 1 (validate every epoch)
- input_modality: use_camera=True
- data_preprocessor: add image normalization (mean/std/bgr_to_rgb)
- train/test pipeline: add multi-view image loading + ImageAug3D + GridMask
- custom_hooks: [] (DisableObjectSampleHook irrelevant for 6-epoch training)
"""

_base_ = [
    './dfbevfusion_lidar_voxel03_second_secfpn_1x4-amp-acc8-cyclic-20e_nus-full-5090.py'
]

backend_args = None
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Redefine db_sampler (inherited value not available as Python variable
# in child config scope).
db_sampler = dict(
    data_root='data/nuscenes/',
    info_path='data/nuscenes/nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5, truck=5, bus=5, trailer=5, construction_vehicle=5,
            traffic_cone=5, barrier=5, motorcycle=5, bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2, truck=3, construction_vehicle=7, bus=4, trailer=6,
        barrier=2, motorcycle=6, bicycle=6, pedestrian=2, traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args))

# --- Model: keep DFBEVFusion, add camera components ---

model = dict(
    type='DFBEVFusion',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        voxelize_cfg=dict(
            max_num_points=32,
            max_voxels=[30000, 40000])),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2),
    fusion_layer=dict(
        type='ConvFuser', in_channels=[80, 256], out_channels=256))

# --- Train pipeline: LiDAR + multi-view camera + ObjectSample ---

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
    dict(type='ObjectSample', db_sampler=db_sampler),
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
    dict(type='GridMask', use_h=True, use_w=True, max_epoch=6,
         rotate=1, offset=False, ratio=0.5, mode=1, prob=0.0,
         fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d',
              'gt_bboxes', 'gt_labels'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
            'cam2lidar', 'ori_lidar2img', 'img_aug_matrix',
            'box_type_3d', 'sample_idx', 'lidar_path', 'img_path',
            'transformation_3d_flow', 'pcd_rotation', 'pcd_scale_factor',
            'pcd_trans', 'img_aug_matrix', 'lidar_aug_matrix',
            'num_pts_feats'])
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
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
            'cam2lidar', 'ori_lidar2img', 'img_aug_matrix',
            'box_type_3d', 'sample_idx', 'lidar_path', 'img_path',
            'num_pts_feats'])
]

# --- Dataloaders ---

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    dataset=dict(
        dataset=dict(
            pipeline=train_pipeline,
            modality=input_modality)))

val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    dataset=dict(
        pipeline=test_pipeline,
        modality=input_modality))
test_dataloader = val_dataloader

# --- Schedule: 6-epoch cosine + momentum (from reference) ---

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=6,
        end=6,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=2.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=2.4,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# --- Optimizer: AMP, lr=2e-4, effective batch 8 ---

optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=4,
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    loss_scale=dict(init_scale=512.0, growth_interval=2000),
    clip_grad=dict(max_norm=35, norm_type=2))

auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/NDS',
        rule='greater'))

# Load LiDAR branch from ep15 checkpoint. Camera branch is initialized
# from img_backbone's init_cfg (ResNet50 from ImageNet); other camera
# components (img_neck, view_transform, fusion_layer) start from scratch.
load_from = 'work_dirs/dfbevfusion_lidar_full_5090/best_NuScenes_NDS_epoch_15.pth'
resume = False

# DisableObjectSampleHook is irrelevant for 6-epoch training (triggers at ep15).
custom_hooks = []
