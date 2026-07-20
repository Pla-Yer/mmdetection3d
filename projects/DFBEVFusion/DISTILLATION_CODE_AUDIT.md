# DFBEVFusion Distillation Code Audit

A code-level review of the LiDAR knowledge distillation implementation.
Every finding references concrete source locations and is categorized by
severity.  This document complements `DISTILLATION_EXPERIMENTS.md`, which
records the experimental outcomes caused by several of the issues below.

## Scope

Reviewed files:

- `projects/DFBEVFusion/bevfusion/distillation.py` — distiller wrapper and
  all KD loss modules.
- `projects/DFBEVFusion/bevfusion/bevfusion.py` — student/teacher feature
  extraction and `extract_distill_features`.
- `projects/DFBEVFusion/bevfusion/transfusion_head.py` — detection head
  forward, loss, and heatmap target generation.
- `projects/DFBEVFusion/bevfusion/middle_adapter.py` — `BEVDownsample`.
- `projects/DFBEVFusion/bevfusion/sparse_encoder.py` — teacher sparse
  middle encoder.
- `projects/DFBEVFusion/configs/dfbevfusion_lidar_distill_*.py` — active
  and ablation configs.

## Critical deficiencies

### C1. Duplicate bbox\_head forward pass

**Location:** `bevfusion/distillation.py:533-580`, `bevfusion/bevfusion.py:325-343`,
`bevfusion/transfusion_head.py:765-783`.

`DFBEVFusionLidarDistiller.loss` calls
`self.student.extract_distill_features`, which internally runs
`self.bbox_head(bev_feat, metas)` (a full TransFusionHead forward: NMS,
top-k proposal selection, Transformer decoder, SeparateHead prediction
layers) solely to extract `dense_heatmap`.  Immediately after,
`self.student.bbox_head.loss(bev_feat, batch_data_samples)` is called,
which internally calls `self(batch_feats, metas)` — running the entire
forward a second time.

The student's detection head forward executes twice per iteration.  The
teacher's detection head also runs a full forward under `torch.no_grad`
(`distillation.py:539`) just to produce `dense_heatmap`, when a direct
call to `self.bbox_head.heatmap_head(fusion_feat.float())` would suffice.

Under AMP fp16, floating-point non-associativity means the two forward
passes can produce slightly different outputs.  The KD target
(`dense_heatmap` from pass 1) and the detection loss target (from pass 2)
are then inconsistent, which is especially problematic for the heatmap
BCE/focal losses where the target distribution matters.

**Recommended fix:** Extract `dense_heatmap` as a side-output of the
detection loss forward.  Either:

1. Have `bbox_head.loss` return `dense_heatmap` alongside the loss dict,
   and remove the separate `bbox_head(...)` call in
   `extract_distill_features`.
2. Or split `bbox_head.forward_single` so that the heatmap branch can be
   called independently when only `dense_heatmap` is needed (teacher
   case).

### C2. `spatial_weight` confidence threshold is a silent no-op

**Location:** `bevfusion/distillation.py:42-49`, configs
`dfbevfusion_lidar_distill_nus-mini-improved-feature.py:30-31` and
`dfbevfusion_lidar_distill_nus-mini.py:44,52,58`.

`_WeightedDistillLoss.spatial_weight` classifies a pixel as foreground
when `teacher_heatmap.sigmoid().amax(dim=1) >= confidence_threshold`
(default `0.1`).  However, `DISTILLATION_EXPERIMENTS.md:477-479` records
that TransFusion's focal-loss dense heatmap has mean teacher GT-center
probability `0.005-0.013`.  A `0.1` threshold rejects virtually every real
foreground pixel.  The weight map degenerates to a uniform
`background_weight=0.1` everywhere, making the object-centric spatial
weighting design a no-op.

This directly affects the currently active middle-feature KD
(`BEVFeatureDistillLoss` with `loss_weight=125`): its spatial weighting is
silently disabled, so the loss is effectively a uniform per-pixel Smooth
L1 over the entire BEV map, dominated by background.

**Recommended fix:** Either lower `confidence_threshold` to match the
focal-loss prior (e.g. `0.005`), or replace the hard threshold with a
soft weighting proportional to `teacher_prob`, or remove
`spatial_weight` entirely from `BEVFeatureDistillLoss` and rely on the
class-aware reliability gate instead.

### C3. `F.normalize` amplifies noise in near-empty BEV cells

**Location:** `bevfusion/distillation.py:65-72` (`BEVFeatureDistillLoss`),
also `:140-142` (`ClassAwareBEVFeatureDistillLoss`),
`:235-237` (`TeacherReliableBEVFeatureDistillLoss`).

Both student and teacher features are channel-normalized to unit length
via `F.normalize(student.float(), dim=1, eps=1e-4)`.  Empty or near-empty
BEV cells (no LiDAR points, near-zero norm) are scaled to unit magnitude
by dividing by `eps=1e-4`.  A noise vector with norm `1e-4` becomes a
unit-norm vector that contributes to the loss equally with high-confidence
cells.  This amplifies background noise and contradicts the intent of
spatial foreground weighting.

**Recommended fix:** Weight each cell's loss contribution by the teacher's
pre-normalization norm (or a sigmoid of it), so that low-energy cells
contribute proportionally less.  Alternatively, mask out cells where the
teacher norm is below a small fraction of the per-batch maximum.

## Moderate deficiencies

### M1. Relation KD deviates from the UniDistill paper

**Location:** `bevfusion/distillation.py:352-377` (`InstanceRelationDistillLoss`).

The code builds a 9x9 cosine relation matrix among 9 sampled points
(center, four corners, four edge midpoints) **within each GT box**.
UniDistill's equations 4-5 build a KxK cosine relation matrix among K
**different objects** (inter-object relations).  `DISTILLATION_EXPERIMENTS.md:56`
claims to match "equations 4-5", but the semantics are different: the
implementation captures intra-box local structure, not inter-object
relations.  Nine points within one small box are spatially close and
highly correlated, so the 9x9 relation matrix carries limited structural
information.

**Recommended fix:** If the intent is to follow UniDistill, reshape the
relation to be across objects: sample one feature vector per GT box
(center point), then compute the KxK cosine matrix across all K boxes in
the batch.  If the intent is intra-box local structure, rename it and do
not claim to match the paper's equations.

### M2. Warmup starts at 50%, not 0%

**Location:** `bevfusion/distillation.py:499-505`.

With the default `warmup_epochs=2`:

| epoch | `_warmup_factor` |
|---:|---:|
| 0 | `(0+1)/2 = 0.5` |
| 1 | `(1+1)/2 = 1.0` |
| 2+ | `min(1.0, 1.5) = 1.0` |

The first epoch already applies 50% KD strength.  This is not a true
warmup from zero.  Additionally, the factor uses epoch count, so it is
constant across all iterations within the same epoch — there is no
smooth per-iteration ramp, unlike the optimizer's 500-iteration
`LinearLR` warmup.

**Recommended fix:** Use `max(0.0, (epoch + 1) / warmup_epochs - 0.5) /
0.5` to start from 0, or switch to an iteration-based ramp that matches
the learning rate warmup window.

### M3. Disabled losses still create graph-connected zero tensors

**Location:** `bevfusion/distillation.py:61` (and similar in all loss
modules), `:548-556`.

When `enabled=False`, each loss returns `student.sum() * 0`.  This
creates a zero tensor that remains connected to the student's autograd
graph.  During `loss.backward()`, autograd traverses the student's graph
through these zero-scaled paths, computing zero gradients unnecessarily.
Furthermore, `bev_loss`, `heatmap_loss`, and `attention_loss` are
computed unconditionally (`:548-556`), while `instance_feature_loss`,
`middle_feature_loss`, `instance_relation_loss`, and
`gaussian_response_loss` are guarded by `is not None` (`:557-579`).  This
inconsistent handling means the first three always pollute the loss dict
with zero tensors even when disabled in the config.

**Recommended fix:** Return a detached zero tensor
(`student.new_zeros(())` or `torch.zeros((), device=student.device)`)
when disabled, or guard the loss computation at the distiller level with
`if self.bev_loss.enabled:`.

### M4. Active middle-feature KD lacks class-aware reliability gating

**Location:** `bevfusion/distillation.py:53-72` (`BEVFeatureDistillLoss`),
configs `dfbevfusion_lidar_distill_nus-mini-improved-feature.py:23-32`.

`DISTILLATION_EXPERIMENTS.md:386-419` documents that the teacher is not
uniformly stronger: bus AP is 0.3336 (teacher) vs 0.4976 (student), a
clear negative-transfer case.  The class-aware variants
(`ClassAwareBEVFeatureDistillLoss`, `TeacherReliableBEVFeatureDistillLoss`)
were designed and tested but abandoned because the exact-box mask and
teacher-correct gate both failed to recover mAP
(`DISTILLATION_EXPERIMENTS.md:442-519`).  However, the default
`improved-feature` config still uses the class-agnostic
`BEVFeatureDistillLoss`, which cannot prevent teacher errors on bus from
contaminating the student.  The negative-transfer problem was never solved
— it was only avoided by disabling middle-feature KD in the full template.

**Recommended fix:** The class-aware gate failed because it only masked
GT box pixels, but teacher errors affect shared convolutional updates and
receptive-field context beyond the box.  A more effective approach would
be to gate by per-class teacher-vs-student AP gap (computed online or
from a held-out set) and apply a per-class scalar loss weight, rather
than a spatial mask.

## Latent bugs (currently disabled, but incorrect if re-enabled)

### L1. `GaussianResponseDistillLoss` coordinate order error

**Location:** `bevfusion/distillation.py:407-409`.

`_foreground_mask` passes `center = torch.stack([center_x, center_y])`
(i.e. `[x, y]`) to `draw_heatmap_gaussian`.  But `draw_heatmap_gaussian`
expects `[row, col]` = `[y, x]` order, as confirmed by
`transfusion_head.py:750` which explicitly passes
`center_int[[1, 0]]` (reversing to `[y, x]`).  The Gaussian mask is drawn
at the transposed position.  Since `x` and `y` grid sizes are equal
(`180 x 180`), no shape error is raised.

### L2. `HeatmapDistillLoss` spatial weight ignores temperature

**Location:** `bevfusion/distillation.py:43` vs `:256`.

`spatial_weight` computes `confidence = teacher_heatmap.sigmoid()` (no
temperature scaling), but the loss target is
`target = (teacher / temperature).sigmoid()`.  The foreground/background
classification is based on unscaled probabilities while the alignment
target uses temperature-scaled probabilities.  A high-temperature setting
would push more pixels above the confidence threshold in the target but
not in the weight map, causing inconsistency.

### L3. `kwargs` silently dropped

**Location:** `bevfusion/distillation.py:533-580`.

`DFBEVFusionLidarDistiller.loss` accepts `**kwargs` but does not forward
them to `self.student.bbox_head.loss(student_outputs['bev_feat'],
batch_data_samples)`.  If `bbox_head.loss` requires any keyword arguments
(e.g. for auxiliary supervision or custom assignment), they are silently
dropped.

## Summary table

| ID | Severity | Location | One-line description |
|---|---|---|---|
| C1 | Critical | `distillation.py:533,545`; `bevfusion.py:331` | bbox\_head forward runs twice per iteration; KD/detection targets may diverge under AMP |
| C2 | Critical | `distillation.py:42-49` | `confidence_threshold=0.1` rejects all real foreground; spatial weighting is a no-op |
| C3 | Critical | `distillation.py:65-72` | `F.normalize` on near-zero BEV cells amplifies noise to unit magnitude |
| M1 | Moderate | `distillation.py:352-377` | 9x9 intra-box relation matrix does not match UniDistill's KxK inter-object relation |
| M2 | Moderate | `distillation.py:499-505` | Warmup starts at 50%, not 0%; epoch-based, no per-iteration ramp |
| M3 | Moderate | `distillation.py:61,548-556` | Disabled losses return graph-connected zero tensors; inconsistent guard logic |
| M4 | Moderate | `distillation.py:53-72`; configs | Active middle-feature KD is class-agnostic; bus negative-transfer never resolved |
| L1 | Latent | `distillation.py:407-409` | `GaussianResponseDistillLoss` passes `[x,y]` to a function expecting `[y,x]` |
| L2 | Latent | `distillation.py:43,256` | `HeatmapDistillLoss` spatial weight uses unscaled sigmoid while target uses temperature-scaled sigmoid |
| L3 | Latent | `distillation.py:533-580` | `**kwargs` accepted but not forwarded to `bbox_head.loss` |

## Relationship to experimental outcomes

The experimental record in `DISTILLATION_EXPERIMENTS.md` is consistent
with the code-level issues identified here:

- **Vanilla KD v1 failure** (`DISTILLATION_EXPERIMENTS.md:42-46`): caused
  by C2 (foreground weighting disabled) and C3 (noise amplification) on
  the full-map BEV loss.
- **Low-level feature KD rejection** (`:107-109`): caused by C3
  (incompatible feature bases amplified by normalization) and C2.
- **Middle-feature KD weight-125 bus regression** (`:375-419`): caused by
  M4 (class-agnostic weighting transfers teacher bus errors) and C2
  (spatial weight is uniform, so bus regions are not downweighted).
- **Teacher-reliable gate failure** (`:490-519`): the gate addressed M4
  at the pixel level but could not prevent shared convolutional updates
  from propagating teacher bus errors through the backbone — the gate
  was too local.
- **Relation KD positive signal** (`:220-248`): relation KD is the only
  objective that does not depend on `spatial_weight` (C2) or
  `F.normalize` (C3), which explains why it is the sole objective that
  survived the matched no-KD control.
