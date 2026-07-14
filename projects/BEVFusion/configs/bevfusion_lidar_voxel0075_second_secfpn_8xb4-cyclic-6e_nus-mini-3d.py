_base_ = ['./bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

data_root = 'data/nuscenes/'
train_ann_file = 'nuscenes_mini_infos_train.pkl'
val_ann_file = 'nuscenes_mini_infos_val.pkl'
metainfo = dict(
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    version='v1.0-mini')

db_sampler = _base_.db_sampler
db_sampler['data_root'] = data_root
db_sampler['info_path'] = data_root + 'nuscenes_mini_dbinfos_train.pkl'

train_pipeline = _base_.train_pipeline
for transform in train_pipeline:
    if transform.get('type') == 'ObjectSample':
        transform['db_sampler'] = db_sampler

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        dataset=dict(
            data_root=data_root,
            ann_file=train_ann_file,
            metainfo=metainfo,
            data_prefix=dict(sweeps='sweeps/LIDAR_TOP'),
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        metainfo=metainfo,
        data_prefix=dict(sweeps='sweeps/LIDAR_TOP')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1))
