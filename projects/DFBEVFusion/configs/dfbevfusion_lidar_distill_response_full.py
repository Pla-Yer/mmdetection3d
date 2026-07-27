"""Full-nuScenes LiDAR response distillation (single-objective).

This is the response-only counterpart to ``dfbevfusion_lidar_distill_
relation_full.py``. It isolates the Gaussian response KD objective on
full nuScenes so the result is directly comparable to the v1 relation-only
run (work_dirs/dfbevfusion_lidar_full_relation_kd/20260720_154637).

Motivation: the relation-only ceiling on this student/teacher pair is
~0.6225 NDS (v1 best 0.6226, v3 saturated at 0.6221). The mini ablation
showed response KD is complementary to relation (geometry vs foreground
confidence) and was the best single-loss mAP contributor on mini. This
run tests whether response alone can match or exceed the relation ceiling
on full nuScenes.

Hyperparameters mirror v1 relation for a fair comparison:

* warmup_epochs=2 (slow ramp; v3 showed fast warmup hurts ep1)
* max_epochs=3 (same as v1)
* lr=1e-5, AdamW, AMP, batch_size=8, seed 20260719

The response loss weight is 10.0, matching the mini response ablation.
The mini joint run measured raw response loss ~0.0366 at weight=10, so
the logged value projects to ~0.37 (about 25% of detection loss
~1.42), a stronger signal than v1 relation's ~5% but appropriate for
a single objective.

All other KD objectives (BEV, heatmap, attention, instance feature,
instance relation) are explicitly disabled.
"""

_base_ = [
    './dfbevfusion_lidar_distill_relation_full.py'
]

# Swap the enabled objective: relation off, response on.
model = dict(
    instance_relation_loss=dict(
        loss_weight=0.0,
        enabled=False),
    gaussian_response_loss=dict(
        loss_weight=10.0,
        enabled=True),
)
