# AnchorDiff algorithm note

## Problem definition

Let (B) be a reference image of a fixed workspace and (F) be a later image
or short capture sequence. The goal is to estimate an object state transition
and a residual visual-change score without treating every changed pixel as
clutter.

## Semantic Change Accounting

1. Estimate a planar transform from optional fixed anchors. Normalize (F) into
   the coordinate system of (B), and retain scale-independent geometry
   evidence. Optional OpenCV adapters detect ArUco or AprilTag markers and
   flatten their corners for the same dependency-free estimator; OpenCV is
   never imported by the CPU core.
2. Compute cheap capture-quality signals: gradient energy as a blur proxy,
   normalized brightness and clipped-pixel ratio.
3. Accept optional object detections from any adapter. Match same-label objects
   with a minimum-cost Hungarian assignment using IoU, center distance and area
   ratio. Emit `stable`, `moved`, `missing` and `new` actions. Stability uses a
   default normalized center-displacement tolerance of `0.1`, which can be
   overridden for a particular camera and scene.
4. Compute a robust difference mask using a median/MAD threshold. Build the
   union of boxes for explained object transitions and remove it from the
   difference mask.
5. Measure residual area, connected components, largest component and spatial
   entropy. Also emit one stable record per residual component with its ID,
   half-open bounding box, pixel area, area ratio and centroid. These are
   evidence, not a hidden embedding.
6. Apply versioned rules to produce a normalized `disorder_score` in `[0, 1]`.
   If geometry, capture quality or detector confidence is unreliable, set
   `needs_review` and leave `disorder_score` unavailable. Preserve the reasons.

Every matched object record includes its action, baseline and final bbox,
displacement, configured tolerance and a short explanation. Every residual bbox
is marked `unexplained_change`, so downstream tools can show why each region is
present. The score is a rule-based comparison signal, not a calibrated
probability or a universal cleanliness scale.

The key invariant is:

```text
changed pixels = explained object change + residual change + capture/unknown
```

The categories are deliberately allowed to be incomplete in v0.1; the system
must preserve unknown evidence rather than silently folding it into a score.

## Normalized homography quality

The estimator reports pixel reprojection RMSE, maximum reprojection error and
the condition number of the 3 x 3 transform. RMSE is divided by the diagonal
of the target anchor extent, producing `normalized_rmse` that is comparable
across image resolutions. A collinear point set, non-finite projection,
ill-conditioned transform or error larger than twice the target extent is
marked `degenerate`; the confidence gate can then return `needs_review`
instead of silently accepting a bad alignment.

Marker adapters return stable `(marker_id, four corners)` observations. Only
IDs present in both images are paired, sorted by ID, and passed to the core as
ordinary 2D point correspondences. This keeps marker detection replaceable and
the evidence contract independent of OpenCV.

## Why this is useful

An ordinary before/after difference fires on camera drift, exposure changes and
a tracked object being moved. SCA makes the object transition explicit, then
scores what remains. That makes the result inspectable and lets a deployment
choose a conservative risk/coverage trade-off.

## Planned evaluation

- object-state macro F1;
- residual-mask IoU and residual-area absolute error;
- rank correlation with a human ordering of workspace disorder;
- risk-coverage curve for `needs_review`;
- latency and peak memory on x86 CPU and RK3588;
- ablations: no alignment, no object explanation and no abstention.
