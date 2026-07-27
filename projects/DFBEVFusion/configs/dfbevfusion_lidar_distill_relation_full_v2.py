"""Full-nuScenes LiDAR relation distillation - aggressive variant v2.

Builds on ``dfbevfusion_lidar_distill_relation_full.py`` and tunes the
relation KD parameters based on the v1 run analysis (work_dirs/
dfbevfusion_lidar_full_relation_kd/20260720_154637):

* v1 baseline: loss_weight=1.0, warmup_epochs=2, max_epochs=3
  -> Best NDS=0.6226 / mAP=0.5327 (vs student 0.6133 / 0.5193)
  -> raw relation loss only dropped 0.084 -> 0.059; KD gradient was
     drowned out by detection losses (~0.64 heatmap, ~0.80 bbox).

* v2 changes (aggressive strategy):
  - loss_weight 1.0 -> 20.0  (~10x stronger KD signal)
  - warmup_epochs=2 -> warmup_iters=500  (full KD strength at iter 500
    instead of 2/3 of training; only ~3% of total iters spent ramping)
  - max_epochs 3 -> 6  (5 epochs at full KD strength instead of 1)
  - cosine LR schedule extended to 6 epochs
  - max_keep_ckpts 3 -> 6  (keep every epoch for inspection)

All other KD objectives remain disabled so the experiment stays
relation-only and directly comparable to v1.
"""

_base_ = [
    './dfbevfusion_lidar_distill_relation_full.py'
]

# Aggressive relation KD: 20x stronger weight, fast warmup, longer run.
model = dict(
    instance_relation_loss=dict(loss_weight=20.0),
    warmup_epochs=0,
    warmup_iters=500,
)

# Cosine LR schedule must match the new max_epochs; the 500-iter linear
# warmup aligns with the KD warmup_iters so LR and KD ramp together.
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
        end=6,
        T_max=6,
        by_epoch=True,
        eta_min_ratio=1e-2,
        convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=6))
