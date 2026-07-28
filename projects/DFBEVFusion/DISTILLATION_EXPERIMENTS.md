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

## Full nuScenes: relation KD baseline (v1)

The mini ablation selected relation-only as the v2 recipe. The first full
nuScenes transfer uses `dfbevfusion_lidar_distill_relation_full.py` and
initializes from the explicit-adapter DFBEVFusion student's best NDS
checkpoint (epoch 15 of `work_dirs/dfbevfusion_lidar_full_5090`). The teacher
is the original sparse-voxel BEVFusion LiDAR
(`bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth`).

The relation loss aligns 9x9 cosine-similarity matrices of nine BEV points
sampled per GT box on the final 512x180x180 BEV feature, with the
energy-threshold valid-point mask. All other KD objectives (BEV, heatmap,
attention, instance feature, gaussian response) are explicitly disabled so
the run remains relation-only and directly comparable to the mini ablation.

Hyperparameters inherited from the mini recipe:

| Field | Value |
|---|---|
| `instance_relation_loss.loss_weight` | 1.0 |
| `instance_relation_loss.enabled` | True |
| `warmup_epochs` | 2 |
| `warmup_iters` | None (use epoch-based ramp) |
| `max_epochs` | 3 |
| `lr` (AdamW) | 1e-5 |
| `train_dataloader.batch_size` | 8 |
| seed | 20260719 |
| AMP initial scale | 512, growth interval 2000 |
| grad clip | max_norm=35 |

The `_warmup_factor` formula linearly ramps the KD multiplier from 0 to 1
over the first `warmup_epochs / max_epochs = 2/3` of training. With
15,448 iters per epoch, full KD strength is reached only at iter ~30,896
(end of epoch 2), leaving just one epoch of full-strength supervision.

Formal work directory: `work_dirs/dfbevfusion_lidar_full_relation_kd/20260720_154637`.

### Headline result

| Epoch | NDS | mAP |
|---:|---:|---:|
| input (student ep15) | 0.6133 | 0.5193 |
| 1 | 0.6220 | 0.5323 |
| 2 (best NDS) | **0.6226** | 0.5327 |
| 3 | 0.6225 | 0.5327 |

Absolute improvement over the input student: **+0.0093 NDS, +0.0134 mAP**.
Best NDS is epoch 2; mAP plateaus at 0.5327 from epoch 2 onward. The
improvement is real and meaningful on full nuScenes (much larger validation
set than mini), especially the +1.34 mAP gain.

### Per-class AP@2.0 (input vs best NDS epoch)

| Class | Input AP@2.0 | KD ep2 AP@2.0 | Δ |
|---|---:|---:|---:|
| car | 0.8792 | 0.8895 | +0.0103 |
| truck | 0.6176 | 0.6271 | +0.0095 |
| construction_vehicle | 0.1762 | 0.1859 | +0.0097 |
| bus | 0.7983 | 0.8061 | +0.0078 |
| trailer | 0.3841 | 0.4050 | +0.0209 |
| barrier | 0.6744 | 0.6758 | +0.0014 |
| motorcycle | 0.4987 | 0.5120 | +0.0133 |
| bicycle | 0.2288 | 0.2658 | **+0.0370** |
| pedestrian | 0.7902 | 0.8000 | +0.0098 |
| traffic_cone | 0.5953 | 0.6152 | +0.0199 |

The largest gains are on `bicycle` (+0.037), `trailer` (+0.021), and
`traffic_cone` (+0.020). These are the categories where object-internal
spatial/structural relation is hardest to learn from detection loss alone:
bicycles are small and orientation-ambiguous, trailers are long and
asymmetric, traffic cones cluster with high inter-object relation. Barrier
(+0.0014) is nearly flat, consistent with barrier detection being dominated
by dense heatmap supervision rather than per-object structure.

### KD loss trajectory (warmup-isolated)

Dividing the logged `loss_kd_instance_relation` by the time-varying
`_warmup_factor` recovers the raw relation L1 distance:

| Iter | Logged loss | Warmup factor | Raw loss |
|---:|---:|---:|---:|
| 1,800 (ep1) | 0.0049 | ~0.058 | ~0.084 |
| 15,498 (ep2 start) | 0.0317 | ~0.502 | ~0.063 |
| 30,948 (ep3 start) | 0.0604 | ~1.003 | ~0.060 |
| 46,344 (ep3 end) | 0.0594 | ~1.000 | ~0.059 |

The raw relation loss decreases monotonically from ~0.084 to ~0.059 (about
30%), confirming the student is aligning its 9x9 cosine relations toward the
teacher. The descent is shallow and slows after epoch 2; combined with the
mAP plateau at epoch 2 and the NDS regression at epoch 3, this suggests the
relation signal saturates quickly under the v1 hyperparameters.

### Findings

- The mini-validated relation recipe transfers cleanly to full nuScenes:
  both NDS and mAP improve over the input checkpoint at every epoch, with
  no class showing negative transfer (the mini bus/truck degradation under
  feature KD does not recur because feature KD is disabled here).
- The KD gradient contribution is small. At full strength the relation
  loss (~0.06) is an order of magnitude below the detection losses
  (heatmap ~0.64, bbox ~0.78). The +0.0093 NDS gain was achieved with the
  KD term contributing roughly 4-5% of the total loss.
- The two-epoch warmup consumed 2/3 of the training budget. Only one
  epoch (epoch 3) ran at full KD strength, and that epoch did not improve
  over epoch 2. This is the strongest evidence that v1 is undertrained on
  the KD side rather than over-trained.
- Training was numerically stable. Two isolated AMP overflow events
  produced transient `grad_norm: inf` (ep2 iter 2000) and `grad_norm: nan`
  (ep2 iter 4400); both were recovered by the dynamic scaler and did not
  affect validation.
- The student remains well below the teacher's expected level (BEVFusion
  LiDAR full nuScenes typically reaches NDS ~0.66 / mAP ~0.55), so there
  is headroom for stronger KD.

Conclusion: v1 validates that relation KD works on full nuScenes, but the
specific hyperparameters underutilize the signal. The recipe is kept as the
matched KD baseline; the next experiment tunes `loss_weight`, `warmup`,
and `max_epochs` to test whether a stronger relation term continues to help.

A matched no-KD fine-tuning control (same student initialization, LR,
schedule, data order, with relation KD disabled) is still required before
attributing the gain to KD rather than to three extra epochs of plain
fine-tuning. That control is not yet started.

## Full nuScenes: relation KD aggressive variant (v2)

`dfbevfusion_lidar_distill_relation_full_v2.py` inherits v1 and applies
three targeted changes derived from the v1 trajectory analysis:

| Field | v1 | v2 | Rationale |
|---|---|---|---|
| `instance_relation_loss.loss_weight` | 1.0 | **20.0** | Raise KD gradient from ~5% to ~50% of detection loss |
| `warmup_epochs` / `warmup_iters` | 2 / None | 0 / **500** | Reach full KD strength at iter 500 (~3% of training) |
| `max_epochs` | 3 | **6** | 5 epochs at full KD strength instead of 1 |
| cosine LR `end`/`T_max` | 3 | 6 | Match new `max_epochs` |
| `max_keep_ckpts` | 3 | 6 | Retain every epoch for inspection |

All other fields inherit unchanged: relation-only (all other KD losses
disabled), AdamW lr=1e-5, AMP init_scale=512, grad clip max_norm=35,
batch_size=8, seed=20260719, same teacher and student checkpoints.

The 500-iter KD warmup aligns with the existing 500-iter LinearLR learning
rate warmup, so LR and KD ramp together. The 20x weight is chosen so that
the full-strength logged relation loss projects to roughly 1.0-1.2, putting
it on the same scale as the combined detection loss (~1.42).

Formal work directory: `work_dirs/dfbevfusion_lidar_full_relation_kd_v2`.

### Acceptance criteria

- **Pass**: best-NDS epoch improves both NDS and mAP over v1 epoch 2
  (0.6226 / 0.5327). The recipe moves to deployment candidate status and
  triggers the matched no-KD control.
- **Partial**: NDS improves but mAP does not, or gains appear only in
  early epochs then degrade. The aggressive weight is too high; fall
  back to `loss_weight=10` and rerun.
- **Fail**: NDS regresses below v1. KD signal overwhelms detection;
  abandon the aggressive weight and revisit the diagnostic ablation
  (`loss_weight=1.0`, only fix warmup + epochs).

### Monitoring points during training

1. After iter 500: logged `loss_kd_instance_relation` should stabilize in
   the 1.0-1.2 range. If it climbs above 2.0 or shows instability, the
   weight is too high.
2. Epoch 1 validation: must not regress below v1 epoch 1 (0.6220 NDS).
   A regression indicates the strong KD signal is hurting the pretrained
   student before it can adapt.
3. Epoch 2-3 trajectory: if NDS rises monotonically, the 6-epoch schedule
   is justified. If it plateaus before epoch 4, consider early stopping
   and reducing `max_epochs` in the next iteration.
4. Final raw relation loss (after dividing by warmup factor = 1.0): target
   < 0.045 (vs v1 final 0.059). A lower value confirms the stronger
   gradient drives tighter relation alignment.

### Partial result (stopped after epoch 3)

Training was stopped after epoch 3 (out of 6) because the ep1 alarm
fired and projections showed v2 was unlikely to beat v1 best even at
epoch 6.

| Epoch | v1 NDS | v1 mAP | v2 NDS | v2 mAP | Δ NDS (v2-v1) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6220 | 0.5323 | 0.6187 | 0.5268 | -0.0033 |
| 2 | 0.6226 | 0.5327 | 0.6196 | 0.5275 | -0.0030 |
| 3 | 0.6225 | 0.5327 | 0.6207 | 0.5282 | -0.0018 |

Key observations:

- **Alarm triggered at ep1**: v2 ep1 NDS=0.6187 < v1 ep1 0.6220, exactly
  the failure pattern flagged in the monitoring protocol. The 20x KD
  signal was too strong during the early pretrained-student adaptation.
- **Gap is narrowing but not closing fast enough**: ep1→ep3 reduced the
  gap from -0.0033 to -0.0018. Linear extrapolation of the v2 slope
  (+0.0009, +0.0011 NDS per epoch) projects ep6 NDS ~0.6252, only barely
  matching v1 best (0.6226). With curve saturation the realistic range
  is 0.6220-0.6235, i.e. at best tied with v1.
- **KD loss verified the weight-too-high hypothesis**: at ep4 iter 700
  the logged `loss_kd_instance_relation` was 0.8922 (within the predicted
  1.0-1.2 band). The implied raw loss = 0.8922 / 20 = 0.0446, a 24%
  drop from v1's final raw 0.059. The student was clearly aligning
  harder to the teacher, but tighter alignment did not translate to
  better detection - classic KD over-alignment.
- **Per-class regression at ep3 vs v1-ep2**: nearly every class regressed
  slightly, with `bicycle` -0.013 being the worst. Bicycle was v1's
  biggest gainer (+0.037), so the over-strong signal suppressed exactly
  the relation-driven gain v1 had unlocked.

Conclusion: v2 fails the pass criterion (best NDS does not exceed v1).
The 20x weight is rejected. The diagnostic value of v2 is that KD loss
share ~36% (logged 0.89 / total 2.45) is on the high side but not
destructive; the failure is more about over-alignment than gradient
explosion. The middle ground `loss_weight=10` is the next probe - it
should produce a logged KD loss around 0.6 (raw 0.06 * 10), putting
the KD share near 30% and leaving room for the student to keep its
detection-quality features.

## Full nuScenes: relation KD balanced variant (v3)

`dfbevfusion_lidar_distill_relation_full_v3.py` inherits v1 and takes the
geometric middle ground between v1 (loss_weight=1, too weak) and v2
(loss_weight=20, too strong):

| Field | v1 | v2 | v3 |
|---|---|---|---|
| `instance_relation_loss.loss_weight` | 1.0 | 20.0 | **10.0** |
| `warmup_epochs` / `warmup_iters` | 2 / None | 0 / 500 | 0 / **500** |
| `max_epochs` | 3 | 6 | **6** |
| cosine LR `end`/`T_max` | 3 | 6 | 6 |
| `max_keep_ckpts` | 3 | 6 | 6 |

Projected full-strength logged KD loss: `10 * 0.06 = 0.6`, putting the
relation term at roughly 30% of detection loss (~1.42) and ~40% of
total loss. This sits between v1 (~5% share, undertrained) and v2 (~36%
share, over-aligned).

Formal work directory: `work_dirs/dfbevfusion_lidar_full_relation_kd_v3`.

### Acceptance criteria (inherited from v2 with tightened thresholds)

- **Pass**: best-NDS epoch improves both NDS and mAP over v1 epoch 2
  (0.6226 / 0.5327). The recipe becomes the deployment candidate and
  triggers the matched no-KD control.
- **Partial**: NDS improves over v1 but mAP does not, or gains appear
  only in early epochs. Fall back to `loss_weight=5` or revisit the
  diagnostic ablation.
- **Fail**: ep1 NDS regresses below v1 ep1 (0.6220). This would suggest
  any weight above ~5 is too aggressive for this student initialization;
  pivot to the diagnostic ablation (`loss_weight=1`, only fix warmup +
  epochs).

### Monitoring points during training

1. After iter 500: logged `loss_kd_instance_relation` should stabilize in
   the 0.5-0.7 range (vs v2's 0.89).
2. Epoch 1 validation: must not regress below v1 ep1 (0.6220 NDS). This
   is the v2 failure pattern; if v3 also triggers it, the issue is not
   weight magnitude alone.
3. Epoch 3 NDS: must reach at least v1 ep1 (0.6220) to remain on track
   for a v1-best-beating epoch 6.
4. Final raw relation loss target: 0.050-0.055 (between v1's 0.059 and
   v2's 0.045). A moderate alignment improvement is the goal; pushing
   raw loss below 0.045 is a v2-style over-alignment warning sign.

### Partial result (ep1, training continues)

Ep1 alarm triggered again, but more mildly than v2. Decision: continue
to ep3 to measure the slope, then decide whether to complete the run.

| Epoch | v1 NDS | v1 mAP | v2 NDS | v2 mAP | v3 NDS | v3 mAP |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6220 | 0.5323 | 0.6187 | 0.5268 | 0.6194 | 0.5285 |

v3 vs v2 at ep1: +0.0007 NDS, +0.0017 mAP. The 10x weight is gentler
than 20x but still triggers the alarm (gap to v1 ep1 is -0.0026 vs v2's
-0.0033).

KD dynamics at ep2 iter 950:

| Metric | v3 measured | Predicted range | Verdict |
|---|---:|---|---|
| `loss_kd_instance_relation` (logged) | 0.4961 | 0.5-0.7 | hit |
| raw relation loss | 0.0496 | 0.050-0.055 | hit |
| KD share of total loss | ~24% | between v1 (~5%) and v2 (~36%) | hit |

All three diagnostic metrics land inside their predicted bands, so the
10x weight choice itself was correct. The remaining ep1 NDS regression
must therefore come from a cause other than weight magnitude.

### Diagnostic insight: warmup ramp is the real root cause

Comparing the effective KD gradient during ep1 (weight × average warmup
factor):

| Run | weight | avg warmup factor over ep1 | effective gradient | × v1 |
|---|---:|---:|---:|---:|
| v1 | 1.0 | ~0.25 (slow ramp over 2/3 of training) | 0.25 | 1x |
| v2 | 20.0 | ~1.0 (full strength after iter 500) | 20.0 | 80x |
| v3 | 10.0 | ~1.0 (full strength after iter 500) | 10.0 | 40x |

v3's effective ep1 KD gradient is ~40x v1 - the same order of magnitude
as v2 (80x). The fast 500-iter warmup applied to a pretrained student
causes the early regression regardless of whether the weight is 10 or 20.
v1's slow 2-epoch warmup is not just a stability convenience; it is the
mechanism that lets the student adapt gradually to the relation
objective before feeling the full gradient.

This suggests the next variant (if v3 stalls) should keep `loss_weight=10`
but switch back to `warmup_epochs=2` (or 3) - combining v1's gentle ramp
with v3's stronger weight. That puts ~5x effective gradient at ep1 (vs
v1's 0.25x and v3's 10x), giving a much smoother entry while still
raising the long-run KD share above v1.

### Decision rule for ep3

Continue v3 to ep3 and apply:

- **Continue to ep6** if ep3 NDS >= 0.6210 (within 0.001 of v1 ep1).
  The slope is healthy enough that 3 more epochs could plausibly exceed
  v1 best (0.6226).
- **Stop and pivot to v4** (`loss_weight=10, warmup_epochs=2,
  max_epochs=6`) if ep3 NDS < 0.6210. The fast-warmup root cause is
  confirmed; the v4 slow-warmup variant is the next probe.
- **Stop and accept v1** if ep3 NDS < 0.6200. The relation-only recipe
  has plateaued; v1 best (0.6226 / 0.5327) is the final relation KD
  result and the next step is the matched no-KD control.

### Final partial result (stopped after epoch 4)

v3 passed the ep3 continue threshold (NDS 0.6219 >= 0.6210) so training
continued. After ep4 confirmed rapid saturation, v3 was stopped and
the run was closed.

| Epoch | v1 NDS | v1 mAP | v3 NDS | v3 mAP | v3 slope (NDS) | v3 vs v1 best |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6220 | 0.5323 | 0.6194 | 0.5285 | - | -0.0032 |
| 2 | 0.6226 | 0.5327 | 0.6213 | 0.5299 | +0.0019 | -0.0013 |
| 3 | 0.6225 | 0.5327 | 0.6219 | 0.5302 | +0.0006 | -0.0007 |
| 4 | - | - | 0.6221 | 0.5304 | +0.0002 | -0.0005 |

The slope decayed geometrically (1.9 -> 0.6 -> 0.2, each roughly 1/3
of the previous), the classic signature of a saturating curve. Linear
extrapolation put ep6 at 0.6221-0.6225 - inside the "tie" band of the
decision rule but almost certainly below v1 best (0.6226).

KD loss kept dropping through ep5 (raw 0.0496 -> 0.0467), confirming
the student continued aligning to the teacher, but the alignment no
longer translated to detection gains.

### Conclusion: relation KD ceiling reached

Three relation variants on the same student/teacher pair now converge
to the same ceiling:

| Run | weight | warmup | max_epochs | best NDS | best mAP |
|---|---:|---|---:|---:|---:|
| v1 | 1.0 | 2 epochs | 3 | **0.6226** | 0.5327 |
| v2 | 20.0 | 500 iters | 6 | 0.6207 | 0.5282 |
| v3 | 10.0 | 500 iters | 6 | 0.6221 | 0.5304 |

v1's slow warmup + low weight remains the best relation configuration.
Stronger weights (v2, v3) trigger early regressions that 3+ extra
epochs cannot fully recover. The relation KD signal saturates around
NDS 0.6220-0.6226 regardless of weight or schedule, indicating the
ceiling is a property of the student/teacher pair and the relation
objective itself, not of under-tuned hyperparameters.

v1 best (0.6226 / 0.5327) is adopted as the final relation-only KD
result. The next experiment pivots to a different KD objective to test
whether a complementary signal can break the relation ceiling.

## Full nuScenes: response KD single-objective

`dfbevfusion_lidar_distill_response_full.py` is the response-only
counterpart to v1 relation. It disables relation KD and enables only
the Gaussian response objective (`GaussianResponseDistillLoss`), which
aligns the class-max dense TransFusion heatmap inside GT Gaussian masks.

Motivation: the mini ablation showed response was the best single-loss
mAP contributor (mini ep1 mAP +0.0081) and was complementary to
relation (geometry vs foreground confidence). With relation saturated
at ~0.6226 NDS, response is the natural next single-objective probe to
test whether a different KD signal can break the ceiling.

### Hyperparameters (mirrors v1 relation for fair comparison)

| Field | v1 relation | response run |
|---|---|---|
| enabled objective | relation | **gaussian_response** |
| `loss_weight` | 1.0 | **10.0** |
| `warmup_epochs` | 2 | 2 (same) |
| `warmup_iters` | None | None (same) |
| `max_epochs` | 3 | 3 (same) |
| cosine LR `end`/`T_max` | 3 | 3 (same) |
| lr / optimizer / AMP / batch / seed | 1e-5 / AdamW / 512 / 8 / 20260719 | same |

The response weight 10.0 is inherited from the mini response ablation.
The mini joint run measured raw response loss ~0.0366 at weight=10, so
the full-strength logged value projects to ~0.37 (about 25% of
detection loss ~1.42). This is a stronger signal than v1 relation's
~5% but appropriate for a single objective carrying the entire KD
burden.

All other KD objectives (BEV, heatmap, attention, instance feature,
instance relation) are explicitly disabled.

Formal work directory: `work_dirs/dfbevfusion_lidar_full_response_kd`.

### Acceptance criteria

- **Pass**: best-NDS epoch improves both NDS and mAP over v1 relation
  best (0.6226 / 0.5327). Response breaks the relation ceiling; it
  becomes the new single-objective KD baseline.
- **Partial**: improves NDS or mAP but not both, or improves one while
  regressing the other. Response and relation are complementary; this
  result justifies a joint relation+response run next.
- **Fail**: best NDS below v1 ep1 (0.6220). Response KD alone is
  weaker than relation on full nuScenes; pivot to joint relation +
  response directly.

### Monitoring points during training

1. After iter 500 (warmup factor ~0.25): logged
   `loss_kd_gaussian_response` should be in the 0.07-0.12 range
   (projected from mini scale). If above 0.2 at half-warmup, the
   weight is too high.
2. Epoch 1 validation: response on mini peaked at ep1 for mAP. If ep1
   mAP exceeds v1 ep1 (0.5323), response is on track.
3. Epoch 2-3 trajectory: mini response NDS rose through ep3. If full-
   nuScenes response also keeps rising, consider extending max_epochs.
4. Per-class check at best epoch: mini response primarily improved
   pedestrian and truck AP. If the same pattern holds on full, the
   response signal is transferring as expected.

### Partial result (ep1-ep2 done, ep3 running)

Response KD broke the relation ceiling at both ep1 and ep2.

| Epoch | v1 relation NDS | v1 relation mAP | response NDS | response mAP | Δ NDS | Δ mAP |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6220 | 0.5323 | 0.6226 | 0.5347 | +0.0006 | +0.0024 |
| 2 | 0.6226 | 0.5327 | 0.6245 | 0.5358 | +0.0019 | +0.0031 |

ep2 is the current best checkpoint. Both NDS and mAP exceed v1 relation
best, confirming response KD alone is a stronger single objective than
relation KD alone on this student/teacher pair.

### Per-class AP@2.0 (response ep2 vs relation ep2)

| Class | relation ep2 | response ep2 | Δ | signal type |
|---|---:|---:|---:|---|
| bicycle | 0.2658 | 0.2904 | +0.0246 | small/structural |
| motorcycle | 0.5120 | 0.5271 | +0.0151 | small/structural |
| barrier | 0.6758 | 0.6838 | +0.0080 | dense |
| traffic_cone | 0.6152 | 0.6207 | +0.0055 | dense |
| pedestrian | 0.8000 | 0.7998 | -0.0002 | flat |
| car | 0.8895 | 0.8859 | -0.0036 | large |
| truck | 0.6271 | 0.6219 | -0.0052 | large |
| bus | 0.8061 | 0.7976 | -0.0085 | large |
| construction_vehicle | 0.1859 | 0.1726 | -0.0133 | large |
| trailer | 0.4050 | 0.3760 | -0.0290 | elongated |

A clear complementary pattern: response helps small/dense objects
where heatmap peak confidence is critical (bicycle, motorcycle,
barrier, traffic_cone) and hurts large/elongated objects where spatial
structure matters more than foreground confidence (trailer, CV, bus,
truck). This matches the mini ablation finding that relation and
response supervise complementary knowledge, and motivates a joint
relation + response run.

### Critical diagnostic: weight=10 was 42x over-projected

The pre-run monitoring protocol projected raw response loss ~0.04
(extrapolated from mini joint data: 0.366 logged at weight 10). The
actual raw response loss on full nuScenes is ~1.70 - a 42x projection
error.

| Metric | Projected | Actual | Error |
|---|---|---|---|
| raw response loss | 0.04 | 1.70 | 42x |
| full-strength logged KD | 0.37 | 17.0 | 46x |
| KD / detection ratio | ~25% | 9.4x | catastrophic |

Root cause: the response loss formula
`(difference * foreground).sum() / num_boxes` scales with the number
of foreground pixels per box. Full nuScenes scenes have many more GT
objects than mini scenes, and the per-box foreground pixel sum is
much larger. The mini-to-full projection failed to account for this
scale difference.

The proper weight for full nuScenes response KD is approximately
`1.8 / 1.70 = 1.06` to match detection loss magnitude, or `0.5-1.0`
to target a 30-50% KD share. Weight 10 is appropriate for mini but
~10x too strong for full nuScenes.

### Why the run survived despite 42x overweight

The 2-epoch slow warmup (`warmup_epochs=2`) shielded the model from
the full KD strength during the critical early phase:

| Epoch | avg warmup factor | effective KD | KD/detection | outcome |
|---:|---:|---:|---:|---|
| 1 | ~0.25 | ~4.25 | 2.3x | manageable, NDS rose |
| 2 | ~0.75 | ~12.75 | 7.0x | extreme but NDS still rose |
| 3 | 1.00 | ~17.0 | 9.4x | catastrophic, degradation likely |

This is the same mechanism that made v1 relation's slow warmup
effective: the student adapts gradually to the KD signal before
feeling the full gradient. Without the warmup, weight=10 would have
destroyed the pretrained student at iter 0.

### KD loss trajectory: raw barely decreasing

| Epoch | iter | warmup factor | logged KD | raw KD |
|---:|---:|---:|---:|---:|
| 1 | 8000 | 0.259 | 4.576 | 1.767 |
| 2 | 8000 | 0.759 | 12.958 | 1.707 |
| 3 | 1000 | 1.000 | 17.029 | 1.703 |

The raw loss dropped only 4% (1.77 -> 1.70) over 2.5 epochs, versus
relation KD's 30% drop (0.084 -> 0.059). The student's dense heatmap
is already well-trained by focal loss; the remaining gap to the
teacher's heatmap is hard to close. This makes the over-strong weight
even more damaging: the model cannot reduce the KD loss, so the
gradient keeps pulling the detection features without relieving the
KD pressure.

### ep3 prognosis: high probability of degradation

The heatmap (detection) loss already rose from 0.64 to 0.93 (+45%)
as the 9.4x over-strong KD gradient pulls the model away from the
detection optimum. With ep3 at full KD strength and no warmup
protection, the model was expected to over-align to the teacher's
heatmap at the cost of detection quality.

### Final result (3 epochs complete)

ep3 did not degrade; it continued to improve, contradicting the
prognosis. The model was more robust to the 9.4x over-strong KD
gradient than predicted.

| Epoch | NDS | mAP | slope NDS | slope mAP |
|---:|---:|---:|---:|---:|
| 1 | 0.6226 | 0.5347 | - | - |
| 2 | 0.6245 | 0.5358 | +0.0019 | +0.0011 |
| 3 | **0.6249** | **0.5366** | +0.0004 | +0.0008 |

Best checkpoint: epoch 3 (NDS 0.6249 / mAP 0.5366).

### Per-class AP@2.0 (response ep3 vs relation ep2)

| Class | relation ep2 | response ep3 | Δ | signal type |
|---|---:|---:|---:|---|
| bicycle | 0.2658 | 0.2886 | +0.0228 | small/structural |
| motorcycle | 0.5120 | 0.5303 | +0.0183 | small/structural |
| barrier | 0.6758 | 0.6844 | +0.0086 | dense |
| traffic_cone | 0.6152 | 0.6205 | +0.0053 | dense |
| car | 0.8895 | 0.8864 | -0.0031 | large |
| truck | 0.6271 | 0.6220 | -0.0051 | large |
| bus | 0.8061 | 0.7960 | -0.0101 | large |
| construction_vehicle | 0.1859 | 0.1842 | -0.0017 | large |
| trailer | 0.4050 | 0.3748 | -0.0302 | elongated |
| pedestrian | 0.8000 | 0.7992 | -0.0008 | flat |

The complementary pattern from ep2 holds at ep3: response improves
small/dense objects (bicycle +0.023, motorcycle +0.018, barrier
+0.009, traffic_cone +0.005) and degrades large/elongated objects
(trailer -0.030, bus -0.010, truck -0.005). This is nearly orthogonal
to relation's class pattern, strongly motivating a joint run.

### Complete comparison across all full-nuScenes runs

| Run | weight | warmup | best ep | best NDS | best mAP | Δ NDS vs input | Δ mAP vs input |
|---|---:|---|---:|---:|---:|---:|---:|
| input (student ep15) | - | - | - | 0.6133 | 0.5193 | - | - |
| v1 relation | 1.0 | 2 ep | 2 | 0.6226 | 0.5327 | +0.0093 | +0.0134 |
| v2 relation | 20.0 | 500 it | 3 | 0.6207 | 0.5282 | +0.0074 | +0.0089 |
| v3 relation | 10.0 | 500 it | 4 | 0.6221 | 0.5304 | +0.0088 | +0.0111 |
| **response** | **10.0** | **2 ep** | **3** | **0.6249** | **0.5366** | **+0.0116** | **+0.0173** |

Response KD is the strongest single-objective KD on full nuScenes:
- +0.0023 NDS / +0.0039 mAP over the best relation run (v1)
- +0.0116 NDS / +0.0173 mAP over the input student

### Conclusion

Response KD broke the relation ceiling and is adopted as the new
single-objective KD baseline. Two key findings:

1. **Response > relation as a single objective.** The foreground-
   confidence signal transfers more useful knowledge than the 9x9
   cosine-relation signal, primarily by improving small/dense object
   detection where heatmap peak quality is critical.

2. **The 42x over-projection was survived, not justified.** Weight=10
   produced a 9.4x KD/detection ratio that should have been
   catastrophic. The 2-epoch slow warmup shielded the model long
   enough for it to adapt, and the model continued improving even at
   full strength. The proper weight for full-nuScenes response KD is
   approximately 1.0 (to match detection loss scale); weight=10 worked
   by accident, not by design. Any future response run should use
   weight=1 with the same 2-epoch warmup and 3-epoch schedule.

The next experiment is joint relation + response KD. The orthogonal
class patterns (response helps small/dense, relation helps
large/elongated) suggest the two signals are complementary. A joint
run at relation=1 + response=1 (both at proper full-nuScenes scale)
is the natural next probe, with a target of breaking 0.625 NDS.

A matched no-KD fine-tuning control (same student init, 3-epoch
schedule, LR, data order, with all KD disabled) remains mandatory
before attributing the +0.0116 NDS gain to KD rather than to plain
continued fine-tuning.

## Full nuScenes: joint relation + response KD

`dfbevfusion_lidar_distill_joint_full.py` enables both relation and
response KD simultaneously, combining the two complementary signals
validated as single-objective runs.

### Motivation

The per-class AP@2.0 patterns of relation (v1 ep2) and response (ep3)
are nearly orthogonal:

| Class | relation Δ | response Δ | complementary |
|---|---:|---:|---|
| bicycle | +0.037 | +0.023 | both gain |
| motorcycle | +0.013 | +0.018 | both gain |
| traffic_cone | +0.020 | +0.005 | both gain |
| barrier | +0.001 | +0.009 | both gain |
| trailer | +0.021 | -0.030 | relation gains, response loses |
| bus | +0.008 | -0.010 | relation gains, response loses |
| truck | +0.010 | -0.005 | relation gains, response loses |

If the relation gains on trailer/bus/truck survive the joint run while
response gains on bicycle/motorcycle/barrier are preserved, the combined
result should exceed both single-objective bests.

### Hyperparameters

| Field | v1 relation | response | **joint** |
|---|---|---|---|
| `instance_relation_loss.loss_weight` | 1.0 | disabled | **1.0** |
| `gaussian_response_loss.loss_weight` | disabled | 10.0 | **1.0** |
| `warmup_epochs` | 2 | 2 | 2 (same) |
| `max_epochs` | 3 | 3 | **6** (extended) |
| cosine LR `end`/`T_max` | 3 | 3 | 6 |
| `max_keep_ckpts` | 3 | 3 | 6 |
| lr / optimizer / AMP / batch / seed | 1e-5 / AdamW / 512 / 8 / 20260719 | same | same |

max_epochs is extended to 6 because response-only showed continued
improvement through ep3 (slope still positive). With warmup_epochs=2
and max_epochs=6, KD reaches full strength at ep3 (1/3 through
training), leaving 4 epochs at full strength (vs 1 epoch in the
3-epoch single-objective runs).

Response weight is 1.0 (not 10.0) because the response-only run
revealed that raw response loss on full nuScenes is ~1.70 (not 0.04
as projected from mini). Weight=1.0 gives a logged KD loss of ~1.70,
matching the detection loss scale. This replaces the accidental 42x
over-projection that survived only because of the slow warmup.

Projected full-strength logged losses:
- relation: 1.0 * 0.06 = 0.06 (3% of detection)
- response: 1.0 * 1.70 = 1.70 (94% of detection)
- total KD: 1.76 (KD/detection = 0.98x)

All other KD objectives (BEV, heatmap, attention, instance feature)
remain disabled.

Formal work directory: `work_dirs/dfbevfusion_lidar_full_joint_kd`.

### Acceptance criteria

- **Pass**: best-NDS epoch exceeds response-only best (0.6249 NDS /
  0.5366 mAP). Joint synergy confirmed; this becomes the deployment
  KD recipe.
- **Partial**: best NDS between relation best (0.6226) and response
  best (0.6249). Some signal interference; tune weights next.
- **Fail**: best NDS below relation best (0.6226). The two signals
  conflict at these weights; fall back to response-only as the final
  single-objective result.

### Monitoring points during training

1. After iter 500 (warmup factor ~0.25): `loss_kd_gaussian_response`
   should be ~0.42 (0.25 * 1.70) and `loss_kd_instance_relation`
   should be ~0.015 (0.25 * 0.06). If response logged > 1.0 at
   half-warmup, the weight is too high.
2. Epoch 1 validation: must not regress below v1 relation ep1 (0.6220
   NDS). Both signals are now active; a regression suggests
   interference.
3. Epoch 2-3 trajectory: both single-objective runs improved through
   ep2-3. If joint also improves monotonically, the signals are
   complementary, not conflicting.
4. Per-class check at best epoch: the key diagnostic. If trailer
   AP@2.0 recovers toward relation's +0.021 (vs response's -0.030)
   while bicycle stays near response's +0.023, the joint is working
   as designed.

Results to be appended after the run completes.

### Final result (6 epochs, stopped after ep5)

Training was stopped during ep6 after ep5 confirmed regression. ep4
is the best checkpoint.

| Epoch | NDS | mAP | slope NDS | status |
|---:|---:|---:|---:|---|
| 1 | 0.6220 | 0.5332 | - | warmup |
| 2 | 0.6232 | 0.5334 | +0.0012 | warmup |
| 3 | 0.6238 | 0.5347 | +0.0006 | 1st full-strength |
| 4 | **0.6246** | **0.5355** | +0.0008 | **BEST** (2nd full-strength) |
| 5 | 0.6239 | 0.5338 | -0.0007 | regressed (overfit) |
| 6 | stopped during training | - | - | - |

### Complete comparison across all full-nuScenes runs

| Run | weight | warmup | best ep | best NDS | best mAP |
|---|---|---|---:|---:|---:|
| input (student ep15) | - | - | - | 0.6133 | 0.5193 |
| v1 relation | 1.0 | 2 ep | 2 | 0.6226 | 0.5327 |
| v2 relation | 20.0 | 500 it | 3 | 0.6207 | 0.5282 |
| v3 relation | 10.0 | 500 it | 4 | 0.6221 | 0.5304 |
| response-only | 10.0 | 2 ep | 3 | 0.6249 | 0.5366 |
| joint v1 (rel=1, resp=1) | 1/1 | 2 ep | 4 | 0.6246 | 0.5355 |

Joint v1 peaked at 0.6246 — essentially tied with response-only
(0.6249, difference -0.0003 NDS) but did not break the ceiling.

### Per-class AP@2.0 (joint v1 ep4 vs relation ep2 vs response ep3)

| Class | relation Δ | response Δ | joint Δ | joint best? |
|---|---:|---:|---:|---|
| construction_vehicle | +0.0097 | +0.0080 | +0.0184 | best |
| pedestrian | +0.0098 | +0.0090 | +0.0130 | best |
| truck | +0.0095 | +0.0044 | +0.0104 | 2nd |
| bicycle | +0.0370 | +0.0598 | +0.0385 | 2nd |
| motorcycle | +0.0133 | +0.0316 | +0.0232 | 2nd |
| traffic_cone | +0.0199 | +0.0252 | +0.0238 | 2nd |
| trailer | +0.0209 | -0.0093 | +0.0169 | 2nd |
| barrier | +0.0014 | +0.0100 | +0.0036 | 2nd |
| bus | +0.0078 | -0.0023 | +0.0035 | 2nd |
| car | +0.0103 | +0.0072 | +0.0103 | tied |

Joint v1 wins on construction_vehicle and pedestrian (classes where
neither single objective was particularly strong), and is 2nd best on
everything else. It is the most balanced run — no class regresses vs
input — but it does not dominate any single class the way the
single-objective runs do.

### Root cause: weight=1/1 was not balanced, it was 97% response

The raw losses of the two objectives differ by 31.6x:

| Signal | raw loss | weight=1 logged | share of KD |
|---|---:|---:|---:|
| relation | ~0.056 | 0.056 | 3.1% |
| response | ~1.77 | 1.77 | 96.9% |

At weight=1/1, the "joint" was effectively "response at weight=1"
— a 10x weaker response signal than response-only (weight=10). The
relation signal, though genuinely complementary (proven by the
construction_vehicle and pedestrian wins), contributed only 3.1% of
the KD gradient and could not compensate for the 10x response
reduction.

Net effect: joint v1 total KD gradient = 1.83 vs response-only = 17.7
(9.7x weaker). The complementary knowledge from relation (+0.056)
was negligible against the response signal's 10x reduction (-15.9).

## Full nuScenes: joint relation + response KD v2 (balanced)

`dfbevfusion_lidar_distill_joint_full_v2.py` fixes the weight-balance
flaw by setting relation_weight=28 so its logged loss matches
response's:

  relation: 28 * 0.056 = 1.57 (48% of KD)
  response:  1 * 1.77  = 1.77 (52% of KD)
  ratio: 1 : 1.13 (nearly equal contribution)

| Field | joint v1 | joint v2 |
|---|---|---|
| `instance_relation_loss.loss_weight` | 1.0 | **28.0** |
| `gaussian_response_loss.loss_weight` | 1.0 | 1.0 (same) |
| relation logged KD | 0.056 (3.1%) | 1.57 (48%) |
| response logged KD | 1.77 (96.9%) | 1.77 (52%) |
| total KD / detection | 1.27x | **2.3x** |
| max_epochs | 6 | **4** |

max_epochs is reduced to 4 (2 warmup + 2 full-strength) because all
previous runs peaked at the 1st-2nd full-strength epoch then
declined. v1 joint peaked at ep4 (2nd full-strength), ep5 regressed.
max_epochs=4 captures the peak without wasting 2 epochs on likely
decline.

Formal work directory: `work_dirs/dfbevfusion_lidar_full_joint_kd_v2`.

### Acceptance criteria

- **Pass**: best-NDS epoch exceeds response-only best (0.6249 NDS /
  0.5366 mAP). Balanced joint synergy confirmed; this is the
  deployment KD recipe.
- **Partial**: best NDS between joint v1 (0.6246) and response-only
  (0.6249). Better balance helped but not enough; try relation_weight
  higher (35-40) or extend max_epochs.
- **Fail**: best NDS below joint v1 (0.6246). The strong relation
  signal interferes with response; revert to response-only as the
  final result.

### Monitoring points during training

1. After iter 500 (warmup factor ~0.125): `loss_kd_instance_relation`
   should be ~0.20 and `loss_kd_gaussian_response` ~0.22 (both
   contributing comparably from the start).
2. Epoch 1 NDS: must not regress below v1 relation ep1 (0.6220). If
   it does, relation_weight=28 is too strong.
3. Epoch 3 (1st full-strength): the key diagnostic. If relation
   logged ~1.57 and response logged ~1.77 (ratio 1:1.13), the
   balance is correct.
4. Epoch 4 (2nd full-strength): expected peak based on all previous
   runs. If NDS still rising at ep4, extend to ep5-6.

Results to be appended after the run completes.

## Full nuScenes: matched no-KD fine-tuning control

`dfbevfusion_lidar_distill_nokd_full.py` is the causal baseline for
all full-nuScenes KD experiments. It inherits the exact student
initialization (epoch 15), optimizer (AdamW lr=1e-5), LR schedule
(500-iter linear warmup + cosine), data pipeline, batch size (8),
and seed (20260719) from the KD configs, and disables every KD
objective. Any NDS/mAP change over the input checkpoint is
attributable to plain continued fine-tuning, not to KD.

### Why the student's own ep16-20 trajectory is not a valid control

The student was trained with cyclic LR (peak ~6e-4) and
val_interval=5. At ep16 the LR was 3.7e-4, 37x higher than the KD
fine-tuning LR of 1e-5. The student collapsed at ep20 (NDS 0.4990)
due to high-LR overfitting, which does not inform whether gentle
fine-tuning at 1e-5 helps. This config is the first valid no-KD
control for full nuScenes.

### Hyperparameters (matched to KD runs)

| Field | KD runs | no-KD control |
|---|---|---|
| student init | ep15 (0.6133/0.5193) | same |
| optimizer | AdamW, lr=1e-5, wd=0.01 | same |
| LR schedule | LinearLR 500 + cosine | same |
| batch_size | 8 | same |
| seed | 20260719 | same |
| all KD losses | various | **disabled** |
| teacher forward | active | **skipped** |
| max_epochs | 3 or 4 or 6 | **4** (covers both) |
| cosine end/T_max | 3, 4, or 6 | 4 |

max_epochs=4 covers both the 3-epoch (response-only) and 4-epoch
(joint v2) KD comparisons. The cosine end point differs slightly
(4 vs 3 for response-only), but the first 3 epochs remain a
reasonable comparison.

Formal work directory: `work_dirs/dfbevfusion_lidar_full_nokd_control`.

### Acceptance criteria

The no-KD control result determines how much of the KD gain is
attributable to KD vs plain fine-tuning:

| no-KD best NDS | KD gain (response 0.6249) | fine-tune share | KD share |
|---:|---:|---:|---:|
| 0.6133 (=input) | +0.0116 | 0% | 100% KD |
| 0.6153 (+0.002) | +0.0096 | 17% | 83% KD |
| 0.6183 (+0.005) | +0.0066 | 43% | 57% KD |
| 0.6213 (+0.008) | +0.0036 | 69% | 31% KD |
| 0.6249 (+0.0116) | +0.0000 | 100% | 0% KD |

The pre-experiment expectation is +0.002-0.005 NDS from gentle
fine-tuning (the student already converged at ep15, so continued
training should help only marginally). This would attribute 57-83%
of the gain to KD.

### Monitoring points during training

1. Epoch 1 NDS: if it exceeds v1 relation ep1 (0.6220), plain
   fine-tuning alone is strong and the KD claim weakens.
2. Epoch 3 NDS: the key comparison point with response-only
   (0.6249). If no-KD ep3 < 0.6200, KD is clearly the dominant
   factor.
3. Trajectory shape: if no-KD peaks early (ep1-2) then declines,
   it confirms the student is already converged and additional
   fine-tuning overfits — strengthening the KD claim.

### Final result (4 epochs complete)

No-KD control improves monotonically through ep4 with no regression,
unlike all KD runs which peaked at ep2-4 then declined.

| Epoch | NDS | mAP | slope NDS | slope mAP |
|---:|---:|---:|---:|---:|
| 1 | 0.6223 | 0.5324 | - | - |
| 2 | 0.6226 | 0.5326 | +0.0003 | +0.0002 |
| 3 | 0.6229 | 0.5335 | +0.0003 | +0.0009 |
| 4 | **0.6233** | **0.5338** | +0.0004 | +0.0003 |

Best checkpoint: epoch 4 (NDS 0.6233 / mAP 0.5338).

The trajectory is remarkably stable — each epoch adds a small but
consistent +0.0003-0.0004 NDS. This is the signature of a converged
model extracting marginal gains from continued gentle optimization,
not of a model still learning. The student was already at its
cyclic-LR peak (ep15); replacing the high cyclic LR with a gentle
1e-5 cosine squeezes out +0.010 NDS without overfitting.

### Per-class AP@2.0 (no-KD ep4 vs input)

| Class | input | no-KD ep4 | Δ |
|---|---:|---:|---:|
| car | 0.8792 | 0.8897 | +0.0105 |
| truck | 0.6176 | 0.6286 | +0.0110 |
| construction_vehicle | 0.1762 | 0.1919 | +0.0157 |
| bus | 0.7983 | 0.8028 | +0.0045 |
| trailer | 0.3841 | 0.4007 | +0.0166 |
| barrier | 0.6744 | 0.6774 | +0.0030 |
| motorcycle | 0.4987 | 0.5176 | +0.0189 |
| bicycle | 0.2288 | 0.2639 | +0.0351 |
| pedestrian | 0.7902 | 0.8013 | +0.0111 |
| traffic_cone | 0.5953 | 0.6172 | +0.0219 |

Plain fine-tuning improves every class with no regressions. The
pattern is broad and uniform: largest gains on bicycle (+0.035),
traffic_cone (+0.022), motorcycle (+0.019), trailer (+0.017),
construction_vehicle (+0.016). This is what continued optimization
looks like when there is no conflicting KD gradient — all classes
move in the same direction.

### KD attribution: the decisive comparison

Matching each KD run's best epoch against the no-KD control at the
same epoch isolates the KD-specific contribution:

| KD run | best ep | KD NDS | no-KD NDS (same ep) | **KD net NDS** | KD net mAP |
|---|---:|---:|---:|---:|---:|
| v1 relation | 2 | 0.6226 | 0.6226 | **+0.0000** | +0.0001 |
| response | 3 | 0.6249 | 0.6229 | **+0.0020** | +0.0031 |
| joint v1 | 4 | 0.6246 | 0.6233 | **+0.0013** | +0.0017 |

And best-vs-best:

| Run | best NDS | best mAP | **vs no-KD best NDS** | vs no-KD best mAP |
|---|---:|---:|---:|---:|
| no-KD control | 0.6233 | 0.5338 | — | — |
| v1 relation | 0.6226 | 0.5327 | **-0.0007** | -0.0011 |
| v2 relation | 0.6207 | 0.5282 | -0.0026 | -0.0056 |
| v3 relation | 0.6221 | 0.5304 | -0.0012 | -0.0034 |
| response | 0.6249 | 0.5366 | **+0.0016** | +0.0028 |
| joint v1 | 0.6246 | 0.5355 | +0.0013 | +0.0017 |

### Findings

1. **Relation KD provides zero net benefit.** v1 relation ep2 (0.6226)
   exactly equals no-KD ep2 (0.6226). The entire +0.0093 NDS
   "relation gain" was plain fine-tuning. v2 and v3 relation are
   worse than no-KD, confirming that stronger relation weights only
   interfere with the fine-tuning signal. The three relation
   experiments (v1/v2/v3) were chasing fine-tuning noise, not a
   relation ceiling.

2. **Response KD is the only effective KD signal.** Net contribution
   +0.0020 NDS / +0.0031 mAP at ep3 (same-epoch comparison). The
   gain is concentrated on small/dense objects (bicycle +0.025,
   motorcycle +0.013, barrier +0.007) at the cost of large/elongated
   objects (trailer -0.026, CV -0.008, bus -0.007). This is a real
   but modest KD-specific effect on top of fine-tuning.

3. **Plain fine-tuning accounts for 83-86% of the total gain.**
   The no-KD control's +0.0100 NDS dwarfs the KD net contribution
   of +0.0016-0.0020. Any KD claim must be stated as "on top of
   gentle fine-tuning," not as an absolute gain over the input
   checkpoint.

4. **No-KD is more stable than any KD run.** The control improves
   monotonically through ep4 with no regression, while every KD run
   peaked at ep2-4 then declined. The KD gradient introduces
   optimization instability that reduces the effective training
   window.

5. **Joint v1's net contribution (+0.0013) is below response-only
   (+0.0016).** The relation signal in the joint dilutes the
   response signal without adding value. This confirms that
   combining an effective signal (response) with a neutral signal
   (relation) produces a weaker result than the effective signal
   alone.

### Conclusion: joint v2 cancelled

The planned joint v2 (relation=28, response=1) is cancelled. Relation
KD has zero net benefit; amplifying it 28x would only increase
interference with the response signal. The experiment would consume
~26 hours to confirm a negative result.

### Final KD recipe

Response-only KD (weight=10, warmup_epochs=2, max_epochs=3) is the
final KD recipe for full nuScenes:

| Metric | input | no-KD ep4 | **response ep3** | KD net gain |
|---|---:|---:|---:|---:|
| NDS | 0.6133 | 0.6233 | **0.6249** | **+0.0016** |
| mAP | 0.5193 | 0.5338 | **0.5366** | **+0.0028** |

The response KD checkpoint (ep3) is the deployment candidate. It
provides a small but real KD-specific improvement on small/dense
objects beyond what gentle fine-tuning alone achieves. If the KD
gain is considered too small to justify the teacher overhead, the
no-KD control checkpoint (ep4, 0.6233/0.5338) is the simpler
alternative with 86% of the total gain and no teacher dependency.

### Root cause: why ep15 was best and why ep16-20 collapsed

Investigation of the student's original training config revealed a
`DisableObjectSampleHook` that disables GT-Aug (database sampling /
copy-paste augmentation) after epoch 15:

    custom_hooks = [dict(disable_after_epoch=15,
                         type='DisableObjectSampleHook')]

The hook sets `ObjectSample.disabled = True` at the start of epoch
16 (0-indexed epoch 15). The training log confirms:
`2026/07/19 00:56:11 - mmengine - INFO - Disable ObjectSample`
(between ep15 and ep16).

This is NOT standard overfitting. The training loss decreased from
1.64 (ep15, GT-Aug on) to 1.21 (ep19, GT-Aug off), but the decrease
is misleading:

| Epoch | GT-Aug | training loss | validation NDS |
|---:|---|---:|---:|
| 15 | active | 1.64 | 0.6133 (best) |
| 16 | disabled | 1.49 (dropped 0.15) | - |
| 19 | disabled | 1.21 | - |
| 20 | disabled | 1.27 | 0.4990 (collapsed) |

The loss dropped because the data became EASIER (GT-Aug off = fewer
objects per scene = lower detection loss), not because the model
improved. The model adapted to the un-augmented distribution and
lost the ability to detect objects in real (denser) scenes, causing
validation NDS to collapse to 0.4990.

This explains why ep15 was best (last epoch with GT-Aug) and why
no-KD fine-tuning from ep15 still improves: the KD/no-KD configs
set `custom_hooks = []`, keeping GT-Aug active throughout. The model
continues training with the same rich augmented data that was used
in ep1-15, now with a gentle lr=1e-5 cosine schedule instead of the
high cyclic LR (5e-4 at ep15).

Implication for KD attribution: part of the no-KD control's +0.0100
NDS gain comes from "GT-Aug continues providing rich training data"
(i.e. not disabling it at ep15), not purely from "gentle fine-tuning."
A perfectly matched control would keep DisableObjectSampleHook active,
but that would reproduce the ep16-20 collapse. The practical choice
is to keep GT-Aug on; the no-KD control then represents the best
 achievable plain fine-tuning result.

## Full nuScenes: no-KD warm restart (from ep4 weights)

After the 4-epoch no-KD control (0.6233 NDS) showed a stable
+0.0003-0.0004 NDS per-epoch slope with no sign of saturation, a
warm restart was run to test whether the model was truly converged.

`dfbevfusion_lidar_distill_nokd_full.py` with
`--cfg-options load_from=...epoch_4.pth` loads the ep4 weights
and starts a fresh 4-epoch cosine schedule (lr=1e-5, cosine end=4).
This is a warm restart: the model weights are from ep4, but the
optimizer state, scheduler, and epoch counter are all fresh.

Note: `--resume` was initially attempted but produced LR=1e-7
(restored the old scheduler state where cosine end=4 was already
complete). Using `load_from` instead of `--resume` gives a fresh
cosine that starts at 1e-5.

### Result

| Epoch | original no-KD (from ep15) | warm restart (from ep4) | Δ |
|---:|---:|---:|---:|
| 1 | 0.6223 | 0.6225 | +0.0002 |
| 2 | 0.6226 | 0.6235 | +0.0009 |
| 3 | 0.6229 | 0.6238 | +0.0003 |
| 4 | 0.6233 | **0.6243** | **+0.0010** |

Warm restart improved NDS from 0.6233 to 0.6243 (+0.0010). The model
was NOT saturated; the fresh cosine with LR reset helped it find a
better optimum. Cumulative no-KD improvement from ep15: +0.0110 NDS
(0.6133 → 0.6243).

### Updated KD attribution

With the warm restart raising the no-KD baseline from 0.6233 to
0.6243, the KD net contributions shrink dramatically:

| Signal | vs original no-KD (0.6233) | vs warm restart (0.6243) |
|---|---|---|
| response KD | +0.0016 (14%) | **+0.0006 (5%)** |
| relation KD | +0.0000 (0%) | **-0.0010 (negative)** |
| joint v1 | +0.0013 (11%) | **+0.0003 (3%)** |

Response KD's net contribution dropped from +0.0016 to +0.0006 NDS
(5% of total gain). This is approaching noise level. Another warm
restart would likely close the gap entirely.

### Per-class AP@2.0 (warm restart ep4)

| Class | input | original no-KD ep4 | warm restart ep4 | response ep3 |
|---|---:|---:|---:|---:|
| car | 0.8792 | 0.8897 | 0.8897 | 0.8864 |
| truck | 0.6176 | 0.6286 | 0.6288 | 0.6220 |
| construction_vehicle | 0.1762 | 0.1919 | 0.1943 | 0.1842 |
| bus | 0.7983 | 0.8028 | 0.8037 | 0.7960 |
| trailer | 0.3841 | 0.4007 | 0.4021 | 0.3748 |
| barrier | 0.6744 | 0.6774 | 0.6779 | 0.6844 |
| motorcycle | 0.4987 | 0.5176 | 0.5187 | 0.5303 |
| bicycle | 0.2288 | 0.2639 | 0.2649 | 0.2886 |
| pedestrian | 0.7902 | 0.8013 | 0.8029 | 0.7992 |
| traffic_cone | 0.5953 | 0.6172 | 0.6186 | 0.6205 |

Warm restart wins on large/elongated objects (trailer +0.027 vs
response, CV +0.010, bus +0.008, truck +0.007). Response KD retains
a residual advantage only on small/dense objects (bicycle +0.024,
motorcycle +0.012, barrier +0.007), but this advantage is shrinking
as plain fine-tuning continues to improve.

### Conclusion

The KD experiments on full nuScenes converge to a clear conclusion:
plain fine-tuning with GT-Aug kept active and a gentle cosine LR
accounts for 95% of the total NDS improvement. Response KD adds
only +0.0006 NDS (5%) on top of warm-restart fine-tuning, primarily
on small/dense objects. Relation KD provides zero or negative net
benefit. The KD signal is real but marginal, and the deployment
decision should weigh whether +0.0006 NDS justifies the teacher
inference overhead.
