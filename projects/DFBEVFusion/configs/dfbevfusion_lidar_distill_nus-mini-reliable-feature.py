"""Selective middle-feature KD on teacher-correct, teacher-better GT regions."""

_base_ = ['./dfbevfusion_lidar_distill_nus-mini-improved-feature.py']

model = dict(
    middle_feature_loss=dict(
        _delete_=True,
        type='TeacherReliableBEVFeatureDistillLoss',
        point_cloud_range=[-54., -54., -5., 54., 54., 3.],
        loss_weight=125.0,
        norm_eps=1e-4,
        # TransFusion dense heatmaps use a low focal-loss prior. Real GT-center
        # probabilities are typically around 0.005--0.013, so 0.1 would reject
        # virtually every object before the rank/correctness gates are applied.
        min_teacher_confidence=0.001,
        box_expand=1.5,
        require_teacher_better=True,
        enabled=True))
