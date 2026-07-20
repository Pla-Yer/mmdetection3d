"""DFBEVFusion LiDAR-only training on full nuScenes with one RTX 5090.

This configuration keeps the validated 0.30 m PointPillars DFBEVFusion
architecture and the original 20-epoch cyclic schedule.  Relative to the
mini configuration it:

1. switches every dataset/database/evaluator input to full nuScenes;
2. raises the hard-voxel capacity from 8k/10k to 30k/40k;
3. uses AMP with four samples per micro-batch;
4. accumulates eight micro-batches to retain the nominal global batch of 32.

If batch_size=4 is not stable in the local software stack, use batch_size=2
and accumulative_counts=16 without changing the learning-rate schedule.
"""

_base_ = [
    './bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# Explicitly override the mini base config. MMEngine recursively merges dicts,
# so omitting version here would retain version='v1.0-mini' from the base.
metainfo = dict(classes=class_names, version='v1.0-trainval')
data_root = 'data/nuscenes/'
backend_args = None

# Preserve more occupied pillars from the 9-sweep point cloud.  Keep
# max_num_points=32 initially: PointPillars PFN memory scales approximately
# with max_voxels * max_num_points, so increasing both at once is expensive
# and makes the source of any accuracy change harder to identify.
model = dict(
    data_preprocessor=dict(
        voxelize_cfg=dict(
            max_num_points=32,
            max_voxels=[30000, 40000])))

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args))

# Redeclare the train pipeline so ObjectSample uses the full-data DB infos
# instead of the mini DB infos captured by the base configuration.
train_pipeline = [
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
        type='GlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix'
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    dataset=dict(
        dataset=dict(
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo)))

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'nuscenes_infos_val.pkl')
test_evaluator = val_evaluator

# One RTX 5090 has enough memory to try a four-sample AMP micro-batch.  Eight
# accumulated micro-batches reproduce the base schedule's nominal batch 32,
# so the inherited 1e-4 -> 1e-3 -> 1e-8 cyclic LR is intentionally unchanged.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=8,
    loss_scale=dict(init_scale=512.0, growth_interval=2000))

auto_scale_lr = dict(enable=False, base_batch_size=32)

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=5)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/NDS',
        rule='greater'))

env_cfg = dict(cudnn_benchmark=True)
randomness = dict(seed=2026, deterministic=False)

load_from = None
resume = False
