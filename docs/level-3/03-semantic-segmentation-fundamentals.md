---
description: "A model predicting 'background' everywhere scored a real 93.75% pixel accuracy -- mean IoU (0.4688) correctly exposed the minority class's IoU of 0.0."
---

# Semantic Segmentation Fundamentals

## What was run

Two real, tiny, hand-verifiable segmentation examples — small enough to check every number by eye — computing
real per-pixel accuracy and real per-class IoU (Intersection over Union) from first principles, to show exactly
why segmentation needs a different real evaluation metric than a simple "percent of pixels correct."

## A real, first example: mostly right, two real pixels wrong

```python
correct = (ground_truth == predicted)
pixel_acc = correct.sum() / correct.size
```
```
REAL per-pixel accuracy: 14/16 = 0.8750
REAL class 0 IoU: 0.7778
REAL class 1 IoU: 0.7778
REAL mean IoU (mIoU): 0.7778
```

With both real classes equally represented (8 pixels each), pixel accuracy and mIoU tell a real, roughly
consistent story here.

## A real, second example that breaks that consistency completely

```python
ground_truth_2[3, 3] = 1  # just one real pixel of the minority class in the whole image
predicted_2 = np.full((4, 4), 0)  # the model predicts "class 0" for literally everything
```
```
real per-pixel accuracy: 0.9375 (looks great!)
real class 0 IoU: 0.9375
real class 1 IoU: 0.0000 (the model never got the minority class right, ever)
real mean IoU: 0.4688 (correctly reveals the real failure)
```

A real, stark demonstration: a model that never once correctly identifies the minority class — getting exactly
`0` real class-1 pixels right, out of the one that exists — still scores a real, misleadingly high `93.75%`
pixel accuracy, simply because the majority class is so dominant. The real mean IoU, by contrast, correctly
collapses to `0.4688`, directly reflecting that one of the two classes was a total failure.

## How It Actually Works

Pixel accuracy counts every real pixel equally and divides by the total, which means a class that covers most
of the image can single-handedly keep the overall score high even if every other class is a complete miss —
exactly what this lesson's second example demonstrates with a real, concrete number. IoU, computed separately
per class as real intersection divided by real union, doesn't get diluted this way: a class with zero correct
pixels gets an IoU of exactly `0.0`, full stop, regardless of how many other pixels elsewhere were classified
correctly. Averaging those per-class IoUs (mean IoU) is what actually surfaces a real, class-specific failure
that a single overall accuracy number can hide completely.

This is the exact same real principle [Level 2, Lesson 9](../level-2/09-class-imbalance-vision.md) demonstrated
for classification — a dominant class can make an aggregate metric look fine while a minority class fails
completely — now shown for segmentation specifically, where the "examples" are individual pixels rather than
whole images, and the imbalance can be even more extreme as a result.

## Common mistakes

- Reporting only pixel accuracy for a segmentation model, especially with class-imbalanced real data — this
  lesson's own real second example shows a `93.75%` pixel accuracy coexisting with a complete, `0.0` IoU
  failure on one class.
- Computing a single, combined IoU across all classes at once instead of per-class IoU averaged afterward —
  this lesson's `class_iou()` function deliberately isolates each real class's intersection and union
  separately, which is what lets a minority class's failure be seen rather than absorbed into the majority
  class's success.
- Assuming a hand-verifiable toy example is "too simple to matter" — this lesson's real 4×4 grids are small
  enough to check by eye specifically so the metric's actual behavior, not just its formula, can be trusted
  before applying it to [Lesson 4](04-running-a-real-segmentation-model.md)'s real, much larger model output.

## Exercise

Build a real third example with three classes instead of two, where the model gets the two majority classes
perfectly right but completely misses a third, small minority class (similar to this lesson's second example).
Compute the real per-class IoU and mean IoU by hand first, then verify your real code's output matches your own
hand computation exactly.
