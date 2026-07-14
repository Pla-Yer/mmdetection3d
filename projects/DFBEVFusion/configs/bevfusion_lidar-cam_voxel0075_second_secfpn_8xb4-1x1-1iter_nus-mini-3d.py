_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-6e_nus-mini-3d.py'
]

model = dict(img_backbone=dict(init_cfg=None))

train_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False)
val_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False)
test_dataloader = val_dataloader

train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop', max_iters=1, val_interval=1)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1))
