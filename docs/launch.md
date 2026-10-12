# AnchorDiff launch brief

## One-line pitch

> Give AnchorDiff a fixed-camera before/after pair; get object state changes,
> unexplained residual regions and a reasoned confidence decision on CPU.

## Audience

- computer-vision engineers who need a transparent change-analysis baseline;
- edge-AI developers targeting mini PCs, N100-class CPUs or RK3588;
- fixed-camera workspaces, inventory checks and inspection workflows;
- researchers who need a deterministic paired-image benchmark instead of a
  black-box workspace score.

## The first demo should show five things

1. a baseline and final image side by side;
2. a moved tracked object with an arrow and displacement;
3. residual regions after the tracked-object region is explained;
4. the JSON evidence and quality signals;
5. a deliberately blurred or misaligned pair that becomes `needs_review`.

This makes the project legible in a short screen recording and demonstrates the
abstention behavior that differentiates it from a plain difference heatmap.

## Release discipline

- Publish the CPU baseline before publishing detector weights.
- Keep all demo images synthetic or explicitly licensed.
- Report the synthetic smoke benchmark as a regression check, never as real
  workspace accuracy.
- Add real-world metrics only after a paired dataset and annotation protocol
  are public.
- Pin model, rules and schema versions in every output file.

## Suggested repository topics

`computer-vision`, `change-detection`, `image-registration`, `edge-ai`,
`explainable-ai`, `opencv`, `workplace-analytics`, `rk3588`, `before-after`.

## Honest success metric

The first goal is not a star count. A successful v0.1 lets a new contributor
clone the repository, run one command, inspect the residual overlay and explain
why the result was accepted or rejected. Stars follow when that loop is useful.
