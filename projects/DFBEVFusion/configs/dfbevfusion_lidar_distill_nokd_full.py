"""Full-nuScenes matched no-KD fine-tuning control.

This is the causal baseline for all full-nuScenes KD experiments. It
inherits the exact student initialization, optimizer, LR schedule,
data pipeline, batch size, and seed from the KD configs, and disables
every KD objective. Any NDS/mAP change over the input checkpoint is
attributable to plain continued fine-tuning, not to KD.

Comparison logic:

  KD gain = KD run best NDS - no-KD control best NDS

If the no-KD control also improves over the input (e.g. +0.003 NDS),
then only the portion above that gain is attributable to KD.

### Why the student's own ep16-20 trajectory is not a valid control

The student was trained with cyclic LR (peak ~6e-4) and val_interval=5.
At ep16 the LR was 3.7e-4, 37x higher than the KD fine-tuning LR of
1e-5. The student collapsed at ep20 (NDS 0.4990) due to high-LR
overfitting, which does not inform whether gentle fine-tuning at 1e-5
helps. This config uses the same gentle 1e-5 cosine schedule as the KD
runs, making it the first valid no-KD control for full nuScenes.

### Schedule: max_epochs=4 (covers both 3-epoch and 4-epoch KD runs)

  ep1-3: comparable to response-only (3-epoch KD, cosine end=3)
  ep1-4: comparable to joint v2 (4-epoch KD, cosine end=4)

Note: the cosine end point differs slightly (4 vs 3 for response-only),
so the LR decline rate is not identical. The first 3 epochs remain a
reasonable comparison because both schedules start at 1e-5 with a
500-iter linear warmup.
"""

_base_ = [
    './dfbevfusion_lidar_distill_relation_full.py'
]

# Disable ALL KD objectives. The distiller wrapper still runs the
# student forward pass and detection losses, but skips the teacher
# forward (need_teacher=False when no KD loss is active).
model = dict(
    warmup_epochs=0,  # no KD warmup needed
    bev_loss=dict(loss_weight=0.0, enabled=False),
    heatmap_loss=dict(loss_weight=0.0, enabled=False),
    attention_loss=dict(loss_weight=0.0, enabled=False),
    instance_feature_loss=dict(loss_weight=0.0, enabled=False),
    instance_relation_loss=dict(loss_weight=0.0, enabled=False),
    gaussian_response_loss=dict(loss_weight=0.0, enabled=False),
)

# 4-epoch cosine LR to cover both 3-epoch and 4-epoch KD comparisons.
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
