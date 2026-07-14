"""Improved masked/local feature-only KD ablation on nuScenes-mini."""

_base_ = ['./dfbevfusion_lidar_distill_nus-mini.py']

# Keep the future matched no-KD control on exactly the same augmentation and
# sampling sequence.
randomness = dict(seed=20260713, deterministic=False)

model = dict(
    # Start from the newly trained explicit-adapter student.  The same
    # checkpoint must be used by the matched no-KD control.
    student_checkpoint=(
        'work_dirs/dfbevfusion_explicit_adapter_20e/epoch_20.pth'),
    # Isolate the redesigned feature term from every previous KD objective.
    bev_loss=dict(enabled=False),
    heatmap_loss=dict(enabled=False),
    attention_loss=dict(enabled=False),
    instance_feature_loss=dict(enabled=False),
    instance_relation_loss=dict(enabled=False),
    gaussian_response_loss=dict(enabled=False),
    # Directly align the explicit student BEVDownsample output with the
    # teacher sparse middle-encoder output (both 256 x 180 x 180).
    middle_feature_loss=dict(
        type='BEVFeatureDistillLoss',
        # Calibrated to make feature KD about 10% of total loss after warm-up.
        # A weight-100 probe measured 4.2% during the epoch-1 0.5 warm-up,
        # projecting to 8.3% at full strength. Weight 125 targets ~10%.
        loss_weight=125.0,
        norm_eps=1e-4,
        confidence_threshold=0.1,
        background_weight=0.1,
        enabled=True))
