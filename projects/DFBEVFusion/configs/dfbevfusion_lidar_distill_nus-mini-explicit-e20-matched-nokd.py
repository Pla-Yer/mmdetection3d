"""Matched no-KD control for the explicit-adapter epoch-20 student."""

_base_ = ['./dfbevfusion_lidar_distill_nus-mini-improved-feature.py']

# The base fixes seed=20260713, the epoch-20 student initialization, LR,
# three-epoch schedule, data pipeline, and teacher. Disable only the remaining
# middle-feature objective; all other KD objectives are already disabled.
model = dict(
    middle_feature_loss=dict(enabled=False))
