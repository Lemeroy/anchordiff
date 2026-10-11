# AnchorDiff algorithm note

## Problem definition

Let (B) be a reference image of a fixed workspace and (F) be a later image
or short capture sequence. The goal is to estimate an object state transition
and a residual visual-change score without treating every changed pixel as
clutter.

## Semantic Change Accounting

1. Estimate a planar transform from optional fixed anchors. Normalize (F) into
   the coordinate system of (B), and retain the reprojection confidence.
2. Compute cheap capture-quality signals: gradient energy as a blur proxy,
   normalized brightness and clipped-pixel ratio.
3. Accept optional object detections from any adapter. Match same-class objects
   with a minimum-cost Hungarian assignment using IoU, center distance and area
   ratio. Emit `returned`, `moved`, `missing` and `new` states.
4. Compute a robust difference mask using a median/MAD threshold. Build the
   union of boxes for explained object transitions and remove it from the
   difference mask.
5. Measure residual area, connected components, largest component and spatial
   entropy. Also emit one stable record per residual component with its ID,
   half-open bounding box, pixel area, area ratio and centroid. These are
   evidence, not a hidden embedding.
6. Apply monotonic, versioned rules. If geometry or quality is unreliable,
   abstain with `needs_review` and preserve the reasons.

The key invariant is:

```text
changed pixels = explained object change + residual change + capture/unknown
```

The categories are deliberately allowed to be incomplete in v0.1; the system
must preserve unknown evidence rather than silently folding it into a score.

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
