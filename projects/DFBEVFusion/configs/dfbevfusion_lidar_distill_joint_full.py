"""Full-nuScenes LiDAR joint relation + response distillation.

Combines the two KD objectives that have been validated as single-objective
runs on full nuScenes:

* relation (v1, ``dfbevfusion_lidar_distill_relation_full.py``):
  best NDS 0.6226 / mAP 0.5327 at weight 1.0, warmup_epochs=2, 3 epochs.
  Strongest gains on bicycle (+0.037), trailer (+0.021), traffic_cone
  (+0.020) — large/elongated objects where spatial structure matters.

* response (``dfbevfusion_lidar_distill_response_full.py``):
  best NDS 0.6249 / mAP 0.5366 at weight 10.0 (accidentally 42x over-
  projected but survived via slow warmup), warmup_epochs=2, 3 epochs.
  Strongest gains on bicycle (+0.023), motorcycle (+0.018), barrier
  (+0.009) — small/dense objects where foreground confidence matters.

The two class-improvement patterns are nearly orthogonal, motivating
this joint run. The target is to break 0.625 NDS by covering both
signal types simultaneously.

Weight rationale: the response raw loss on full nuScenes is ~1.70
(measured during the response-only run), so weight=1.0 produces a
logged KD loss comparable to the detection loss (~1.8). This is the
proper scale, replacing the accidental weight=10 that produced a 9.4x
KD/detection ratio. Relation raw loss is ~0.06, so weight=1.0 keeps
it at the proven v1 level (~3% of detection). The two signals together
produce ~1.76 logged KD, roughly matching the detection loss.

All other KD objectives (BEV, heatmap, attention, instance feature)
remain disabled. The schedule uses warmup_epochs=2 (reaches full KD
strength at ep3, leaving 4 epochs at full strength) and max_epochs=6
(extended from the single-objective 3 epochs, since response showed
continued improvement through ep3 and the joint signal may need more
time to converge).
"""

_base_ = [
    './dfbevfusion_lidar_distill_relation_full.py'
]

# Enable both relation and response at proper full-nuScenes scale.
model = dict(
    instance_relation_loss=dict(
        loss_weight=1.0,
        enabled=True),
    gaussian_response_loss=dict(
        loss_weight=1.0,
        enabled=True),
)

# Extend to 6 epochs; cosine LR must match. The 500-iter linear LR
# warmup and the 2-epoch KD warmup are inherited unchanged.
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
