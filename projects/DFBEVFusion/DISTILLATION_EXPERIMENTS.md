# DFBEVFusion LiDAR Distillation Experiments

## Goal

Transfer detection knowledge from the original sparse-voxel BEVFusion LiDAR
model to the deployment-oriented PointPillars DFBEVFusion model. The distilled
LiDAR checkpoint will later initialize LiDAR-camera fusion training.

## Baselines on nuScenes-mini

| Model | Epoch | NDS | mAP |
|---|---:|---:|---:|
| BEVFusion teacher | 15 | 0.2804 | 0.2108 |
| DFBEVFusion student | 15 | 0.2456 | 0.1890 |

Both checkpoints use the same LiDAR-only nuScenes-mini split. Their common
post-middle-encoder path produces a single 512 x 180 x 180 BEV feature map and
a 10 x 180 x 180 dense TransFusion heatmap.

## Vanilla KD v1

The first implementation combined:

- channel-normalized full-map BEV Smooth L1;
- temperature-scaled soft-target heatmap BCE;
- full-map activation attention MSE;
- a two-epoch linear KD warm-up.

The student was initialized from epoch 15 and fine-tuned with AdamW. An AMP
overflow caused by normalizing empty BEV cells with an excessively small
epsilon was fixed by using `norm_eps=1e-4`. The stable run used an initial
learning rate of `2e-5`.

| Distillation epoch | NDS | mAP |
|---:|---:|---:|
| 1 | 0.2490 | 0.1894 |
| 2 | 0.2486 | 0.1879 |
| 3 | 0.2498 | 0.1832 |
| 4 | 0.2376 | 0.1879 |
| 5 | 0.2399 | 0.1868 |

Conclusion: v1 did not produce a meaningful accuracy gain. Its best NDS gain
was within mini-set variance and accompanied by lower mAP. Full-map averaging
was dominated by background, soft BCE had a non-zero entropy floor, and the
attention loss was approximately 1e-6. Training was stopped after epoch 5 and
v1 is retained only as the vanilla KD baseline.

## Object-centric KD v2 (paper/code audited)

The second design was revised after checking the UniDistill paper equations,
ablations, and the SparseKD CenterHead logit-KD implementation:

- sample the first SECOND stage (low-level BEV) at nine points per augmented
  GT box and apply direct L1, matching UniDistill equation 3;
- sample the post-neck high-level BEV and align the 9 x 9 cosine relation
  matrix with L1, matching equations 4-5;
- sigmoid the dense TransFusion classification heatmap, gather its class-wise
  maximum response, and apply L1 only inside CenterPoint-style GT Gaussian
  masks, following equations 6-7 and the classification-max ablation;
- start from the LiDAR-student weights reported for UniDistill's
  strong-teacher path: feature 10, relation 1, response 10. These are a paper
  reproduction starting point, not assumed optimal for three-epoch
  same-modality fine-tuning;
- disable all v1 full-map losses by default;
- fine-tune from the same student checkpoint at `1e-5` with two-epoch warm-up.

The adaptive 1x1 layers described by UniDistill are not used because they are
introduced when the teacher is weaker than the student. Here the teacher is
measurably stronger and both selected feature maps have matching channels and
resolution. SparseKD's teacher-confidence, foreground, and rank masks are not
mixed into the default UniDistill run; they remain a separate future logit-KD
ablation because TransFusion does not expose CenterHead-style dense regression
maps.

References:

- [UniDistill paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_UniDistill_A_Universal_Cross-Modality_Knowledge_Distillation_Framework_for_3D_Object_CVPR_2023_paper.pdf)

- [UniDistill official repository](https://github.com/megvii-research/CVPR2023-UniDistill)
- [SparseKD paper and official repository](https://github.com/CVMI-Lab/SparseKD)

The v2 experiment must use a new work directory and must be compared against a
non-KD fine-tuning run with the same initialization, learning rate, epochs, and
data order. A gain is accepted only if both NDS and mAP improve beyond normal
mini-set variation.

Before the joint run, use the per-loss `enabled`/`loss_weight` fields to run
feature-only, relation-only, and response-only ablations. On one real mini
sample the joint weighted losses were approximately 3.81, 0.10, and 0.70,
respectively, so the paper feature weight is comparatively strong for this
fine-tuning setup and must be judged by validation metrics rather than copied
silently.

AMP uses an explicit dynamic scaler initialized at 512 with a 2000-step growth
interval. The default PyTorch scale of 65536 intermittently overflowed through
the low-level point-sampling backward path and contaminated each ten-step
`grad_norm` logging window with `inf`.

## Object-centric KD v2 ablations

All three runs start from the same DFBEVFusion epoch-15 checkpoint, use the
same three-epoch schedule and validate after every epoch. The original student
baseline is NDS 0.2456 / mAP 0.1890.

| Distillation | Epoch | NDS | mAP | mATE | mASE | mAOE | mAVE |
|---|---:|---:|---:|---:|---:|---:|---:|
| feature | 1 | 0.2291 | 0.1799 | 0.6268 | 0.5583 | 1.2322 | 1.2922 |
| feature | 2 | 0.2414 | 0.1823 | 0.5546 | 0.5343 | 1.1841 | 1.1303 |
| feature | 3 | 0.2441 | 0.1788 | 0.5483 | 0.5139 | 1.1104 | 1.1375 |
| relation | 1 | 0.2560 | 0.1867 | 0.5122 | 0.5104 | 1.2363 | 1.2672 |
| relation | 2 | 0.2499 | 0.1895 | 0.5379 | 0.5199 | 1.1981 | 1.2348 |
| relation | 3 | 0.2540 | 0.1923 | 0.5200 | 0.5143 | 1.2384 | 1.1175 |
| response | 1 | 0.2402 | 0.1970 | 0.6114 | 0.5673 | 1.1759 | 1.1741 |
| response | 2 | 0.2366 | 0.1875 | 0.6149 | 0.5809 | 1.2469 | 1.3208 |
| response | 3 | 0.2539 | 0.1910 | 0.5209 | 0.5145 | 1.2153 | 1.2396 |

### Findings

- Low-level feature L1 is rejected: it never exceeds the original student and
  ends with lower NDS and mAP. Although the low-level maps have matching shape,
  sparse-voxel and pillar encoders produce incompatible local feature bases;
  direct point-wise alignment causes negative transfer.
- Relation KD is effective. Epoch 3 improves NDS by about 0.0084 and mAP by
  about 0.0034. It also improves velocity error substantially, while its best
  NDS occurs at epoch 1.
- Response KD is effective but noisier on mini. Epoch 1 gives the best mAP
  (about +0.0081), while epoch 3 gives a balanced gain of about +0.0083 NDS
  and +0.0021 mAP.
- Relation and response supervise complementary knowledge: geometry/structure
  versus foreground confidence. The next experiment is their joint loss from
  the original student initialization. Feature KD is disabled by default and
  must not be included in the joint run.

The joint run is accepted only if its epoch-selected checkpoint improves both
NDS and mAP over the baseline. On mini, report both the best-NDS and best-mAP
epochs rather than selecting a checkpoint from training loss.

## Relation + response joint result

| Epoch | NDS | mAP | mATE | mASE | mAOE | mAVE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2519 | 0.1909 | 0.5354 | 0.5179 | 1.1505 | 1.2021 |
| 2 | 0.2543 | 0.1916 | 0.5170 | 0.5220 | 1.1412 | 1.1614 |
| 3 | 0.2390 | 0.1928 | 0.6155 | 0.5739 | 1.2305 | 1.2576 |

Epoch 2 is the joint checkpoint selected by NDS. Relative to the original
student it gains about 0.0087 NDS and 0.0026 mAP, but it only exceeds the best
single relation NDS by about 0.0003 and has lower mAP. Epoch 3 has the highest
joint mAP but sharply degrades localization, scale, orientation, velocity, and
NDS, so it is rejected.

The joint experiment confirms that relation and response supervision are both
useful, but does not demonstrate a meaningful synergy on nuScenes-mini. The
third epoch is overfitting. For a deployment-oriented LiDAR initialization,
use joint epoch 2 when prioritizing NDS, or relation epoch 3 when preferring the
more balanced NDS/mAP pair. Do not use joint epoch 3.

Further weight tuning on the same mini split risks selecting validation noise.
The next meaningful experiment is either a multi-seed confirmation or moving
the relation + response recipe to full nuScenes with a full-data teacher and a
matched no-KD fine-tuning control.

## Follow-up mini weight balance

The joint `relation=1 / response=10` logs show the following average weighted
losses:

| Epoch | Relation | Response | Detection | Response / relation |
|---:|---:|---:|---:|---:|
| 1 | 0.0504 | 0.3660 | 1.9874 | 7.26 |
| 2 | 0.0986 | 0.6740 | 1.9662 | 6.84 |
| 3 | 0.0981 | 0.6653 | 1.9706 | 6.78 |

Response supervision is consistently about seven times the relation term.
Two targeted follow-ups are defined while holding every other setting fixed:

- `r1-r5`: relation 1, response 5, expected ratio around 3.4;
- `r1-r2`: relation 1, response 2, expected ratio around 1.4.

These are diagnostic mini experiments, not a broad hyperparameter search. Both
must start from the original student epoch-15 checkpoint and a fresh work
directory. Compare epoch-wise NDS/mAP and error metrics with `r1-r10`, relation
only, and the no-KD control.

### Follow-up results

| Setting | Epoch | NDS | mAP | mATE | mASE | mAOE | mAVE |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1-r5 | 1 | 0.2350 | 0.1875 | 0.6105 | 0.5852 | 1.1430 | 1.2381 |
| r1-r5 | 2 | 0.2479 | 0.1871 | 0.5266 | 0.5436 | 1.2468 | 1.2732 |
| r1-r5 | 3 | 0.2391 | 0.1920 | 0.6012 | 0.5812 | 1.1678 | 1.2214 |
| r1-r2 | 1 | 0.2505 | 0.1925 | 0.5314 | 0.5211 | 1.2018 | 1.2041 |
| r1-r2 | 2 | 0.2389 | 0.1895 | 0.6115 | 0.5712 | 1.1935 | 1.2698 |
| r1-r2 | 3 | 0.2404 | 0.1908 | 0.6090 | 0.5667 | 1.2394 | 1.1603 |

The weighted response averages were approximately 0.35 for r1-r5 and 0.15 for
r1-r2, versus a relation average of 0.10. Even near loss-scale balance, r1-r2
degrades after epoch 1. Therefore the failure is not explained by response
weight magnitude alone.

The current class-max response discards category identity and can reinforce a
teacher peak even when its predicted class differs from the GT/student class.
It also overlaps with the student's supervised focal heatmap objective. This
can help classification in isolation but conflict with relation KD when both
change the same high-level representation. Weight search over 10/5/2 is closed.

The selected v2 recipe is relation-only, epoch 3 (NDS 0.2540 / mAP 0.1923).
Both low-level feature KD and class-max Gaussian response KD are disabled in
the default mini and full templates. Before attributing the gain to KD, a
matched no-KD fine-tuning control remains mandatory. If response KD is revisited,
it should be redesigned as class-aware SparseKD-style heatmap alignment with
teacher-confidence/foreground masks, not another scalar weight adjustment.

The matched control configuration is
`dfbevfusion_lidar_distill_nus-mini-nokd.py`. It retains the identical student
initialization, optimizer, schedule, data pipeline, and distiller wrapper while
setting every KD loss to zero. Its validation metrics are the required causal
baseline for the relation-only result.

## Matched no-KD control result

| Epoch | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2382 | 0.1917 | 0.6074 | 0.5696 | 1.2067 | 1.2015 | 0.4001 |
| 2 | 0.2370 | 0.1864 | 0.6072 | 0.5735 | 1.2228 | 1.1738 | 0.3806 |
| 3 | 0.2492 | 0.1865 | 0.5267 | 0.5185 | 1.2320 | 1.2813 | 0.3947 |

The fairest epoch-3 comparison is:

| Model | NDS | mAP | mATE | mASE | mAOE | mAVE |
|---|---:|---:|---:|---:|---:|---:|
| no-KD control | 0.2492 | 0.1865 | 0.5267 | 0.5185 | 1.2320 | 1.2813 |
| relation KD | 0.2540 | 0.1923 | 0.5200 | 0.5143 | 1.2384 | 1.1175 |
| relation minus control | +0.0048 | +0.0059 | -0.0067 | -0.0042 | +0.0064 | -0.1638 |

Plain fine-tuning explains part of the NDS gain over the original epoch-15
student, but it lowers mAP. Relation KD exceeds the matched control in both NDS
and mAP, slightly improves translation and scale, and substantially improves
velocity error. Orientation is marginally worse. This pattern is consistent
with relation KD transferring object-internal BEV structure and motion-related
context rather than merely extending optimization.

The mini conclusion is therefore narrowed to one supported claim: relation KD
has a positive signal beyond matched fine-tuning. Low-level feature KD and the
current class-max response KD are rejected. Because this is one run on an
81-sample validation split, the magnitude is not yet statistically stable; use
relation epoch 3 as the selected mini checkpoint, then confirm with additional
seeds or full nuScenes before making a final accuracy claim.

## Feature-distillation redesign research

The failed feature experiment rejects only raw L1 at nine GT points on the
first shared SECOND stage. It does not show that encoder knowledge is
untransferable. The teacher/student architecture actually diverges before that
stage:

- teacher sparse middle encoder output: approximately 256 x 180 x 180;
- student PointPillars scatter output: 256 x 360 x 360;
- both are then consumed by related SECOND/FPN detection paths.

The student already performs the necessary spatial adaptation in the first
SECOND stage: its first stride-2 Conv2d maps the 256 x 360 x 360 scatter map to
a 128 x 180 x 180 feature. The teacher uses stride 1 at the corresponding stage
because its sparse middle encoder already outputs at 180 x 180. The failed
nine-point feature experiment sampled these post-convolution 128 x 180 x 180
maps, so adding another stride-2 adapter would duplicate an operation already
present in the deployment network.

The remaining mismatch is representational, not spatial. If a learnable
adapter is tested, it should be a training-only 1 x 1 or 3 x 3 stride-1
128-to-128 projection on the student's first SECOND output, optionally with
normalization, before masked/local-structure KD. Alternatively, distill only
local affinity and channel statistics at this point and avoid raw values
entirely.

### Explicit deployment downsample revision

The deployment student was subsequently refactored to make the downsampling
boundary explicit:

`PointPillarsScatter(256x360x360) -> BEVDownsample(256x180x180) -> SECOND[1,2]`.

Previously the first SECOND stage used stride 2 directly. The new adapter is a
3x3 stride-2 Conv2d followed by BN and ReLU, while SECOND now matches the
teacher's `[1,2]` stage strides. Its convolution is initialized as center-sample
channel identity and BN as identity, rather than random initialization. Old
student checkpoints load all existing parameters strictly while allowing only
the newly introduced adapter keys to be absent.

This is a real student architecture change. Earlier no-KD/KD metrics belong to
the old fused-downsample architecture and cannot serve as the final baseline
for the revised model. The revised student requires a new matched no-KD
fine-tuning baseline before any new feature-distillation claim.

PointDistiller shows that direct backbone imitation is suboptimal for sparse
point clouds. Its effective design selects teacher-important voxels by
channel-wise maximum response, gathers K-nearest local neighborhoods, encodes
their geometry with dynamic graph convolution, and reweights learning toward
important/multi-point voxels. SparseKD similarly restricts feature/logit KD to
teacher-confidence, foreground and rank-selected positions. UniDistill uses a
learnable adaptation layer when feature bases are not safely compatible.

For this codebase, the practical staged approximation is:

1. reuse the existing stride-2 first SECOND stage and, if needed, learn only a
   stride-1 128-to-128 feature-basis adapter after it;
2. build a mask from GT foreground union teacher top-N activation positions;
3. align normalized adapted features with cosine/Smooth-L1 only under the mask;
4. align 3 x 3 local affinity matrices (center-to-neighbor cosine), transferring
   local geometry without requiring channel values to be identical;
5. gate each GT region by teacher confidence at its class center, so weak or
   incorrect teacher regions do not supervise the student;
6. retain the already validated high-level relation KD as a separate term.

Before a full training run, run two diagnostics. First, freeze both networks
and train only the adapter on cached/online encoder features; a clearly falling
held-out reconstruction/cosine loss establishes representational learnability.
Second, compare teacher and student validation quality per class and at GT
centers; feature KD should be enabled only for classes/regions where the teacher
has a real advantage. The mini teacher leads by only about 0.035 NDS and 0.022
mAP overall, so ungated imitation can transfer teacher errors. A stronger full
nuScenes teacher is expected to make feature KD more reliable.

References:

- [PointDistiller paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_PointDistiller_Structured_Knowledge_Distillation_Towards_Efficient_and_Compact_3D_Detection_CVPR_2023_paper.pdf)
- [PointDistiller official repository](https://github.com/RunpeiDong/PointDistiller)
- [SparseKD paper/repository](https://github.com/CVMI-Lab/SparseKD)
- [UniDistill paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_UniDistill_A_Universal_Cross-Modality_Knowledge_Distillation_Framework_for_3D_Object_CVPR_2023_paper.pdf)

## Explicit BEVDownsample feature KD ablation

The corrected redesigned feature-only run uses
`dfbevfusion_lidar_distill_nus-mini-improved-feature.py` and starts from the
already trained explicit-adapter student's epoch-20 checkpoint (NDS 0.2114 /
mAP 0.1772). It is KD fine-tuning, not training the base student from scratch.

The feature boundary is now exposed explicitly:

`student PointPillarsScatter -> BEVDownsample (256x180x180)`

is aligned directly with:

`teacher sparse middle encoder (256x180x180)`.

No additional projection or deployment layer is introduced. The loss uses
FP32 channel normalization and confidence-weighted Smooth L1. All final-BEV,
heatmap, attention, old post-SECOND nine-point feature, relation, and response
KD terms are disabled so this remains a strict feature-only ablation. The
aborted `distill_improved_feature_explicit_e20` run used the wrong post-SECOND
128-channel feature boundary and must not be reported as an experiment result.

An initial middle-feature run with weight 10 was also stopped during epoch 1
as a loss-scale diagnostic. Across 54 log windows its KD term averaged 0.0114,
only 0.5% of total loss during the 0.5 warm-up (about 0.0228 / 1% projected at
full strength). A subsequent weight-25 scale check produced about 0.0285
during the 0.5 warm-up, projecting
to only about 2.5% of total loss at full strength. It was stopped at epoch 1,
iter 210. The requested formal target is approximately 10% of total loss, so
the next scale probe used weight 100. Its first logs measured 4.2% during
warm-up, which projects to 8.3% at full strength. Solving
`KD / (detection + KD) = 10%` from those measured values gives approximately
125, which is the final formal-run weight.

### Weight-125 result

| Epoch | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| input | 0.2114 | 0.1772 | 0.6070 | 0.5922 | 1.2400 | 1.0788 | 0.5732 |
| 1 | **0.2337** | 0.1694 | 0.5708 | **0.5109** | 1.3114 | 1.2313 | **0.4280** |
| 2 | 0.2331 | **0.1702** | **0.5627** | 0.5183 | 1.2815 | 1.1608 | 0.4385 |
| 3 | 0.2298 | 0.1652 | 0.5715 | 0.5211 | **1.2702** | **1.1369** | 0.4355 |

The loss was numerically stable with no NaN/Inf gradient logs. Mean feature-KD
loss was 0.1425 in the warm-up epoch and 0.2831/0.2848 afterward. As detection
loss decreased, the measured KD share reached 12.6% rather than the requested
10%; a fixed coefficient cannot hold a constant ratio throughout training.

The best NDS checkpoint is epoch 1 and the best distilled mAP checkpoint is
epoch 2. Compared with the input checkpoint, every epoch improves NDS but
reduces mAP. Pedestrian and truck AP improve, while bus AP falls by 0.11-0.16
and dominates the overall mAP regression. This is not a successful joint
NDS/mAP result. A matched no-KD fine-tune from the same explicit-adapter epoch
20 checkpoint and seed is still required before assigning the NDS gain to KD.

### Teacher per-class transfer diagnosis

The epoch-15 teacher is not uniformly stronger than the explicit-adapter
student. Mean AP over the four nuScenes distance thresholds is:

| Class | Teacher | Student input | KD epoch 1 | KD minus input |
|---|---:|---:|---:|---:|
| car | 0.6247 | 0.4975 | 0.4980 | +0.0005 |
| truck | 0.2796 | 0.2075 | 0.2265 | +0.0190 |
| bus | 0.3336 | **0.4976** | 0.3647 | **-0.1329** |
| motorcycle | 0.0565 | 0.0431 | 0.0448 | +0.0018 |
| pedestrian | 0.8139 | 0.5266 | 0.5596 | +0.0330 |

Construction vehicle, trailer, and barrier have no GT instances in this mini
validation split and therefore cannot be assessed. Bicycle (52 instances) and
traffic cone (39 instances) both have teacher AP 0 and are additional observed
teacher weaknesses, although the student also has AP 0 for them.

Across these five non-zero-AP classes, the teacher-student gap and the epoch-1
KD change have Pearson correlation about 0.86 (n=5, trend evidence only). Bus
is the clearest negative-transfer case: at the 1 m threshold the teacher,
student, and KD epoch-1 AP values are 0.1585, 0.5270, and 0.1675 respectively.
The distilled student moves almost directly toward the weaker teacher. Its
mean bus AP closes about 81% of the student-to-teacher gap in one epoch.

This strongly supports the interpretation that the corrected feature KD is
transferring useful teacher representation for pedestrian/truck while also
transferring a poor bus representation. It is not yet strict causal proof:
mini has only 41 bus validation instances and a matched no-KD run is missing.
The current class-agnostic spatial weighting uses the maximum teacher heatmap
confidence across classes, so it cannot reject a bus location that the teacher
confidently represents as the wrong class. A class-aware teacher-reliability
gate or bus-region downweight ablation is more appropriate than further global
loss-weight tuning.

### No-bus feature KD ablation protocol

`dfbevfusion_lidar_distill_nus-mini-improved-feature-no-bus.py` tests the bus
negative-transfer hypothesis before running the matched no-KD control. It
keeps the epoch-20 explicit-adapter initialization, seed 20260713, three-epoch
schedule, weight 125, warm-up, teacher, and every detection loss identical to
the weight-125 feature-KD run.

The only change is a class-aware spatial reliability map: feature-KD weight is
zero inside each augmented GT bus BEV box and one for all other GT classes.
The ordinary supervised detection losses remain active for bus. Overlapping
regions use the minimum reliability weight so a bus region cannot be restored
by another box. Two mask unit tests and one real-data AMP optimization step
passed with finite gradients before the full run.

Acceptance criterion: compare epoch-wise bus AP, overall mAP, and NDS directly
with weight-125 feature KD. If bus AP and mAP recover without erasing the NDS
gain, proceed to the matched no-KD run. If not, merely masking the exact GT box
is insufficient; teacher errors may affect surrounding receptive-field context
or other categories, and broader/class-confidence gating should be tested.

### No-bus feature KD result

| Epoch | No-bus NDS | Original KD NDS | ΔNDS | No-bus mAP | Original KD mAP | ΔmAP |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2341 | 0.2337 | +0.0004 | 0.1696 | 0.1694 | +0.0002 |
| 2 | 0.2334 | 0.2331 | +0.0003 | 0.1706 | 0.1702 | +0.0004 |
| 3 | 0.2295 | 0.2298 | -0.0003 | 0.1656 | 0.1652 | +0.0004 |

Bus AP for no-bus/original KD is 0.3639/0.3647 at epoch 1,
0.3846/0.3832 at epoch 2, and 0.3394/0.3396 at epoch 3. These differences
are negligible, and all distilled mAP values remain below the input student's
0.1772. The exact-box mask therefore fails the acceptance criterion and the
requested matched no-KD run is not started.

The result rules out a purely local explanation: negative transfer is not
confined to direct feature matching at pixels inside bus GT boxes. Shared
encoder updates, context outside boxes, downstream receptive fields, or plain
fine-tuning may dominate. KD remained numerically stable and its loss share was
effectively unchanged (12.6-12.7% after warm-up), so the null result is not due
to a collapsed global KD term. A broader GT-class-aware teacher correctness
gate is the next KD design option; scientifically, matched no-KD remains the
cleanest way to separate KD effects from fine-tuning effects.

### Teacher-reliable feature KD protocol

`dfbevfusion_lidar_distill_nus-mini-reliable-feature.py` replaces the failed
exact bus-box exclusion with positive selection. A GT object contributes
middle-feature KD only when the teacher dense heatmap ranks its GT class first,
the teacher GT-class probability exceeds 0.001, and that probability is higher
than the student's. KD is restricted to a 1.5x expanded rotated GT box and is
weighted by the teacher GT-class probability. All other foreground and all
background receive zero feature-KD weight; supervised detection losses remain
unchanged for every class, including bus.

The small absolute threshold is deliberate: five real training batches showed
that TransFusion's focal-loss dense heatmap has mean teacher GT-center
probability 0.005-0.013. A generic 0.1 threshold rejected virtually every
object. With 0.001, the reliability gate retained 6-11 correct-and-better
objects per batch (49-513 BEV pixels). The epoch-1 half-warm-up KD loss was
stable at 0.142-0.155 with gradient norm 5.74-7.12, projecting to roughly a
10-12% full-strength loss share. Four mask/gating unit tests and five real AMP
optimization steps passed before the formal run.

The controlled constants remain the same as the weight-125 and no-bus runs:
explicit-adapter epoch-20 initialization, teacher checkpoint, seed 20260713,
three epochs, LR, warm-up, data order, and feature weight 125. The formal work
directory is `work_dirs/distill_reliable_middle_feature_w125_explicit_e20`.

### Teacher-reliable feature KD result

| Epoch | Reliable NDS | Original KD NDS | Reliable mAP | Original KD mAP |
|---:|---:|---:|---:|---:|
| 1 | 0.2342 | 0.2337 | 0.1699 | 0.1694 |
| 2 | 0.2335 | 0.2331 | **0.1706** | 0.1702 |
| 3 | 0.2300 | 0.2298 | 0.1655 | 0.1652 |

The reliable gate produces only +0.0002 to +0.0005 NDS and +0.0003 to
+0.0005 mAP over original KD. It is also effectively tied with the exact
no-bus mask. Every epoch remains below the input student's 0.1772 mAP, so the
mAP-recovery acceptance criterion fails.

At the best-mAP epoch 2, reliable KD obtains car/truck/bus/motorcycle/pedestrian
mean AP of 0.4954/0.2160/0.3865/0.0498/0.5588. Relative to the input student,
bus still loses 0.1111 AP, while pedestrian and truck gain 0.0322 and 0.0085.
Bus 1 m AP is 0.2748 versus the input's 0.5270. The result therefore preserves
the same classwise transfer pattern instead of correcting it.

Optimization was stable. Mean middle-KD loss was 0.1528 in the half-warm-up
epoch and 0.3052/0.3062 afterward; its full-strength share was 13.45-13.51%.
No training loss, KD loss, or gradient NaN/Inf was found. The gate's failure is
not loss collapse or numerical instability.

Conclusion: teacher-wrong GT pixels are not the sole source of the bus drop.
Shared convolutional updates, cross-class/context effects, and ordinary
fine-tuning remain plausible. Under the user's stated rule, the matched no-KD
run is not started because absolute mAP did not improve. Scientifically, that
control is now more informative than another spatial mask if the goal changes
to separating KD effects from fine-tuning drift.

### Matched no-KD control protocol

After reviewing the reliable-KD result, the matched no-KD control was
authorized. `dfbevfusion_lidar_distill_nus-mini-explicit-e20-matched-nokd.py`
inherits the exact weight-125 experiment setup: explicit-adapter epoch-20
initialization, seed 20260713, data order and augmentation, batch size, LR,
three-epoch cosine schedule, validation cadence, and detection losses. The
only experimental change is `middle_feature_loss.enabled=False`; every other
KD objective was already disabled. Its isolated work directory is
`work_dirs/distill_explicit_e20_matched_nokd`.

This control separates continued fine-tuning from feature KD. Compare each
epoch directly with original, no-bus, and reliable KD using NDS, mAP, bus AP,
pedestrian AP, and truck AP. If no-KD reproduces the bus decline and NDS gain,
fine-tuning dominates. If KD runs diverge materially from no-KD, the difference
is attributable to the feature objective under the matched setup.
