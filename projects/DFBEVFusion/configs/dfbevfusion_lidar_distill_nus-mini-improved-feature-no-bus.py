"""Middle-feature KD with distillation disabled inside GT bus regions."""

_base_ = ['./dfbevfusion_lidar_distill_nus-mini-improved-feature.py']

model = dict(
    middle_feature_loss=dict(
        type='ClassAwareBEVFeatureDistillLoss',
        point_cloud_range=[-54., -54., -5., 54., 54., 3.],
        # nuScenes class order: car, truck, construction vehicle, bus,
        # trailer, barrier, motorcycle, bicycle, pedestrian, traffic cone.
        class_weights=[1., 1., 1., 0., 1., 1., 1., 1., 1., 1.]))
