---
description: "A real DeepLabV3 labeled all 270,400 pixels as 'person' or 'background' -- 43.92% person -- in 0.82s, its mask extent tracking Lesson 2's detector box."
---

# Running a Real Segmentation Model

## What was run

A real, pretrained DeepLabV3 (ResNet-50 backbone, trained on 21 real PASCAL VOC classes) run on the same real
astronaut photograph from [Lesson 2](02-running-a-real-detector.md), producing a real per-pixel class map
rather than a bounding box, compared directly against that lesson's real detection result.

## A real, dense, per-pixel prediction

```python
output = model(img_tensor.unsqueeze(0))["out"]
pred_mask = output.argmax(1).squeeze().numpy()
```
```
real output shape: (1, 21, 520, 520) (batch, 21 classes, H, W)
```
```
=== REAL per-pixel class assignment ===
  __background__ : 151649 pixels (56.08%)
  person         : 118751 pixels (43.92%)
```

Every single one of the real 270,400 pixels in the resized image received exactly one of the 21 real class
labels — and for this image, the model only ever used two of them, splitting the image almost evenly between
`person` (`43.92%`) and `__background__` (`56.08%`).

## A real, direct comparison against the detector's box

```python
rows_with_person = np.where(person_mask.any(axis=1))[0]
cols_with_person = np.where(person_mask.any(axis=0))[0]
```
```
real person-mask bounding extent: rows [17, 519], cols [0, 452]
(for reference, Lesson 2's real Faster R-CNN person BOX was (12,18,390,506))
```

**Honest caveat:** the two real models resize their input differently (DeepLabV3 to `520×520`, Faster R-CNN to
`512×512`), so these coordinates aren't on an identical pixel grid — a small, real scale difference (about
1.6%) exists between them. Even accounting for that, the real segmentation mask extends further left (to
column `0`) and further down (to row `519`) than the detector's box, suggesting the pixel-level mask captured
some real person-shaped content near the image edges that the detector's single rectangular box didn't fully
enclose.

## How It Actually Works

A detector like [Lesson 2](02-running-a-real-detector.md)'s Faster R-CNN answers "where, roughly, is an object
of this class" with a single rectangle — convenient, but necessarily including some real background pixels
inside that rectangle, and potentially missing real object pixels that fall outside it if the object's true
shape isn't rectangular. A segmentation model like this lesson's DeepLabV3 instead answers "what class is
*this exact pixel*" for every pixel independently, which is why its real mask can trace a real, non-rectangular
silhouette more closely than any bounding box could — and why this lesson's mask extent differs, even if only
modestly, from the detector's box on the same real underlying person.

## Common mistakes

- Treating a bounding box and a segmentation mask as interchangeable, equally precise descriptions of "where
  an object is" — this lesson's own real comparison shows they can disagree at the edges, because a rectangle
  structurally cannot trace a non-rectangular real silhouette.
- Comparing two different models' pixel coordinates without checking whether they preprocess (resize) input
  the same way — this lesson's real `520×520` vs `512×512` resize difference is a small but genuine source of
  coordinate mismatch that has to be accounted for, not ignored.
- Assuming a segmentation model's class list is as broad as a detection model's — this lesson's real DeepLabV3
  only has 21 real PASCAL VOC classes, far fewer than [Lesson 2](02-running-a-real-detector.md)'s 91 real COCO
  classes; a real pipeline needing both broad class coverage and pixel-level precision has to pick models that
  actually support both.

## Exercise

Re-run this lesson's exact real segmentation on `skimage.data.coffee()` (a real photograph containing a cup —
not one of DeepLabV3's 21 real VOC classes) and check what the model predicts for every real pixel when the
actual content has no matching class at all, connecting directly to the same real "forced wrong answer"
pattern [Level 2, Lesson 5](../level-2/05-cnn-architectures-resnet.md) found for out-of-distribution
classification.
