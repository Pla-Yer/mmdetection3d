"""DFBEVFusion LiDAR-camera warm restart from ep6.

Extends training from the 6-epoch LiDAR+Cam run (best ep6, NDS 0.6816)
with a fresh 6-epoch cosine schedule. The model weights are loaded
from epoch_6.pth via ``load_from`` (not ``--resume``) to get a fresh
optimizer state and scheduler, avoiding the LR=1e-7 dead-end that
``--resume`` would produce.

The LR, momentum, and schedule are identical to the original 6-epoch
run, giving the model a fresh cosine from 2e-4 down to 2e-8. This is
a standard warm restart: the fresh LR burst may help escape local
optima that the declining cosine at the end of the original run
could not.

Usage:
  python tools/train.py <this_config> \
    --work-dir work_dirs/dfbevfusion_lidar_cam_resnet50_warmrestart
"""

_base_ = [
    './dfbevfusion_lidar-cam_voxel03_second_secfpn_resnet50_nus-full-5090.py'
]

# Fresh 6-epoch cosine from ep6 weights. The inherited param_scheduler
# (cosine end=6, T_max=6 + momentum) is correct as-is for a warm restart.
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=6,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/NDS',
        rule='greater'))

# Load ep6 weights (best from original run). Fresh optimizer + scheduler.
load_from = 'work_dirs/dfbevfusion_lidar_cam_resnet50/epoch_6.pth'
resume = False
