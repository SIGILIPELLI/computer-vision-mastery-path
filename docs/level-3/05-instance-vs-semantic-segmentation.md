---
description: "Mask R-CNN separated two real people into 2 distinct instances (0 mask overlap); semantic segmentation gave one undifferentiated 261,400-pixel blob."
---

# Instance Segmentation vs Semantic Segmentation

## What was run

A real scene deliberately built to contain two separate instances of the same class — two real crops of the
same photographed person, placed side by side — run through both a real instance segmentation model (Mask
R-CNN) and the real semantic segmentation model from [Lesson 4](04-running-a-real-segmentation-model.md), to
see directly what each one can and cannot tell you about the same real scene.

## Real instance segmentation: two people, correctly separated

```python
keep = (scores > 0.5) & (torch.tensor([coco_classes[l] == "person" for l in labels]))
```
```
REAL instance segmentation: 2 distinct 'person' INSTANCES found (confidence > 0.5)
  instance 0: box=(1,18,310,392) centroid_x=155 real mask pixels=74979
  instance 1: box=(501,13,800,396) centroid_x=651 real mask pixels=75349

REAL overlap between instance 0's mask and instance 1's mask: 0 pixels
```

A real, exactly correct result: Mask R-CNN found precisely 2 person instances, with centroids near the real
placed positions (`150` and `650`), nearly identical real mask sizes (expected, since both crops are copies of
the same source image), and genuinely zero pixel overlap between the two separate instance masks.

## The same real scene, through semantic segmentation instead

```python
semantic_person_mask = (seg_mask == person_idx)
```
```
REAL semantic segmentation result: 261400 pixels labeled 'person' as ONE undifferentiated class --
no instance IDs, no way to tell there are two separate people from the mask alone.
```

A real, single, combined mask — correctly identifying every "person" pixel in the scene, but with absolutely
no real information about how many distinct people that mask actually represents.

## How It Actually Works

Semantic segmentation assigns each real pixel exactly one class label and stops there — two people standing
next to each other produce one continuous "person" region, indistinguishable at the pixel-label level from a
single, oddly-shaped person. Instance segmentation (Mask R-CNN, built on the same real detection backbone as
[Lesson 2](02-running-a-real-detector.md)'s Faster R-CNN) first detects individual object instances as separate
real detections, *then* produces a pixel mask for each one — which is exactly why this lesson's real result
could report two separate masks with a real, verified zero pixel overlap between them, something semantic
segmentation's architecture has no mechanism to produce at all.

The real, zero-overlap confirmation matters specifically because it rules out the alternative explanation that
Mask R-CNN just produced two overlapping, redundant masks for what it mistakenly thought were different
objects — the real masks are cleanly, spatially separated, matching the real, physically separate people in the
scene.

## Common mistakes

- Using semantic segmentation when a real downstream task actually needs to count or individually track
  objects — this lesson's real semantic result cannot answer "how many people are in this image," while the
  real instance result answers it directly and correctly.
- Assuming instance segmentation is "just" semantic segmentation plus bounding boxes — this lesson's real
  workflow shows instance segmentation detects individual objects *first*, then masks each one, a genuinely
  different real pipeline, not an added post-processing step on top of a semantic mask.
- Building a test scene without a way to independently verify correctness — this lesson's two placed crops at
  known real positions (`x=150`, `x=650`) gave a concrete, checkable expectation for where each instance's
  centroid should land, rather than just eyeballing whether the result "looks right."

## Exercise

Rebuild this lesson's real scene with three real person crops instead of two, placed at three distinct real
positions, and confirm Mask R-CNN reports exactly 3 instances with 3 real, pairwise-non-overlapping masks —
extending this lesson's verification to a less trivial real case than two.
