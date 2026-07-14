"""Mini ablation: relation weight 1 plus response weight 5."""

_base_ = ['./dfbevfusion_lidar_distill_nus-mini.py']

model = dict(
    instance_feature_loss=dict(enabled=False),
    instance_relation_loss=dict(loss_weight=1.0, enabled=True),
    gaussian_response_loss=dict(loss_weight=5.0, enabled=True))

