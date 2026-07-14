"""Matched three-epoch fine-tuning control with all KD losses disabled."""

_base_ = ['./dfbevfusion_lidar_distill_nus-mini.py']

model = dict(
    bev_loss=dict(enabled=False),
    heatmap_loss=dict(enabled=False),
    attention_loss=dict(enabled=False),
    instance_feature_loss=dict(enabled=False),
    instance_relation_loss=dict(enabled=False),
    gaussian_response_loss=dict(enabled=False))

