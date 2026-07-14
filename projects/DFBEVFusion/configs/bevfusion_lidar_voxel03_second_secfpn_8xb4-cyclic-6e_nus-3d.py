_base_ = ['./bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

param_scheduler = [
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

train_dataloader = dict(batch_size=2, num_workers=2, persistent_workers=True)
val_dataloader = dict(batch_size=2, num_workers=2, persistent_workers=True)
test_dataloader = dict(batch_size=2, num_workers=2, persistent_workers=True)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1))
