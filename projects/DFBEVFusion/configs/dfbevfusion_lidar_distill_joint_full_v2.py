"""Full-nuScenes LiDAR joint relation + response distillation v2.

Builds on ``dfbevfusion_lidar_distill_joint_full.py`` and fixes the
weight-balance flaw that prevented v1 from exceeding response-only.

### Root cause of v1's failure to break response-only

v1 used relation=1, response=1. But the raw losses differ by 31.6x:

  relation raw loss  ~0.056
  response raw loss  ~1.77

At weight=1/1, response contributed 96.9% of the KD gradient and
relation only 3.1%. The "joint" was effectively "response at weight=1"
(i.e. 10x weaker than response-only at weight=10). Relation's real but
tiny 3.1% contribution could not compensate for the 10x response
reduction. v1 peaked at 0.6246 vs response-only 0.6249.

### v2 fix: balance logged losses by raw-loss inverse ratio

  relation_weight = 28  -> logged = 28 * 0.056 = 1.57
  response_weight = 1  -> logged = 1  * 1.77  = 1.77
  ratio = 1 : 1.13  (nearly equal contribution)

Total KD ~3.34, KD/detection ~2.3x (healthy, vs v1's 1.27x and
response-only's 12.3x). Both signals now carry meaningful gradient.

### Schedule: max_epochs=4 (2 warmup + 2 full-strength)

All previous runs peaked at the 1st-2nd full-strength epoch then
declined (v1 joint: ep4 peak, ep5 regress; response-only: ep3 peak).
With warmup_epochs=2, full strength starts at ep3. max_epochs=4 gives
2 full-strength epochs, enough to capture the peak without wasting
time on likely-declining ep5-6. If the curve is still rising at ep4,
extend.

All other KD objectives remain disabled. lr/optimizer/AMP/batch/seed
unchanged from v1 relation.
"""

_base_ = [
    './dfbevfusion_lidar_distill_relation_full.py'
]

# Balanced joint: relation 28x to match response's 31.6x larger raw loss.
model = dict(
    instance_relation_loss=dict(
        loss_weight=28.0,
        enabled=True),
    gaussian_response_loss=dict(
        loss_weight=1.0,
        enabled=True),
)

# 4 epochs: 2 warmup + 2 full-strength. Cosine LR matches.
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
        end=4,
        T_max=4,
        by_epoch=True,
        eta_min_ratio=1e-2,
        convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=4, val_interval=1)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=4))
