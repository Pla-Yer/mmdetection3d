"""Full-nuScenes no-KD control extended to 8 epochs.

Resumes from the 4-epoch no-KD control checkpoint
(work_dirs/dfbevfusion_lidar_full_nokd_control/epoch_4.pth, NDS
0.6233) and extends training to 8 epochs to test whether plain
fine-tuning continues to improve.

Motivation: the 4-epoch no-KD control showed a stable +0.0003-0.0004
NDS per epoch slope with no sign of saturation. If this continues,
no-KD ep8 would project to ~0.6247 NDS — within 0.0002 of response
KD best (0.6249). This would further shrink the KD net contribution.

The cosine LR schedule is extended from end=4 to end=8. At the resume
point (ep5), the LR will be ~3.2e-6 (vs the original ep4 LR of 1e-7
which was essentially zero). This gives the model enough gradient
to continue learning while staying gentle.

Usage: --resume from the same work_dir to load ep4 checkpoint and
continue from ep5.
"""

_base_ = [
    './dfbevfusion_lidar_distill_nokd_full.py'
]

# Extend to 8 epochs (4 more beyond the original 4).
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
        end=8,
        T_max=8,
        by_epoch=True,
        eta_min_ratio=1e-2,
        convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=1)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=8))
