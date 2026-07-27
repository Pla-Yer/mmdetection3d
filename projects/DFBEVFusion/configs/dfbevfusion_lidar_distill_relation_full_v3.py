"""Full-nuScenes LiDAR relation distillation - balanced variant v3.

Builds on ``dfbevfusion_lidar_distill_relation_full.py`` and applies the
lesson learned from v2 (loss_weight=20 was too aggressive; v2 never matched
v1 at any epoch and triggered the ep1 regression alarm).

v3 takes the geometric middle ground between v1 (loss_weight=1, too weak)
and v2 (loss_weight=20, too strong):

* loss_weight 1.0 -> 10.0  (~5x stronger KD signal than v1, ~0.5x of v2)
* warmup_epochs=2 -> warmup_iters=500  (inherited from v2; reaches full
  strength at iter 500 instead of 2/3 of training)
* max_epochs 3 -> 6  (inherited from v2; 5 epochs at full KD strength)
* cosine LR schedule extended to 6 epochs
* max_keep_ckpts 3 -> 6

Projected full-strength KD loss: 10 * 0.06 = 0.6, putting the relation
term at roughly 40% of detection loss (~1.42). This is below v2's 36%
share but well above v1's ~5%.

All other KD objectives remain disabled so the experiment stays
relation-only and directly comparable to v1 and v2.
"""

_base_ = [
    './dfbevfusion_lidar_distill_relation_full.py'
]

# Balanced relation KD: 10x weight (between v1's 1x and v2's 20x).
model = dict(
    instance_relation_loss=dict(loss_weight=10.0),
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
