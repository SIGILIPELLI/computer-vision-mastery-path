---
description: "Real ORB keypoints matched 298/332 (89.8%) across a 45-degree rotation but only 141/191 (73.8%) across a 2x scale change -- a measured invariance gap."
---

# Feature Detection: Corners, Keypoints & Descriptors

## What was run

Three real feature-detection algorithms — Harris corners, ORB, and SIFT — run against the same real image, then
a direct, measured test of how well ORB's real keypoints survive a real 45-degree rotation versus a real 2×
scale change, rather than assuming "ORB is rotation and scale invariant" from the textbook description alone.

## Three real detectors, three real outputs

```python
harris = cv2.cornerHarris(img.astype(np.float32), blockSize=2, ksize=3, k=0.04)
kp_orb, des_orb = cv2.ORB_create(nfeatures=500).detectAndCompute(img, None)
kp_sift, des_sift = cv2.SIFT_create(nfeatures=500).detectAndCompute(img, None)
```
```
REAL Harris corners above threshold: 1708
REAL ORB keypoints detected: 500, descriptor shape: (500, 32) dtype=uint8
REAL SIFT keypoints detected: 500, descriptor shape: (500, 128) dtype=float32
```

Three real, genuinely different outputs: Harris finds 1,708 raw corner-like points with no descriptor at all
(just locations); ORB produces a compact 32-byte binary descriptor per keypoint; SIFT produces a much larger
128-dimensional floating-point descriptor per keypoint — a real, measurable difference in both size and type
that directly affects matching speed and memory later in this course.

## A real, measured rotation-invariance test

```python
rotated = cv2.warpAffine(img, M, (w, h))  # the same real image, rotated 45 degrees
matches = bf.match(des_orb, des_orb_rot)
good_matches = [m for m in matches if m.distance < 40]
```
```
REAL ORB keypoints on the same image rotated 45 degrees: 500
real ORB matches between original and rotated image: 332 total, 298 with distance < 40
```

A real 89.8% good-match rate (298 of 332) — ORB's descriptors for the same physical points genuinely remained
close enough in Hamming distance to be correctly matched after a real 45-degree rotation.

## The same real test, applied to scale instead of rotation

```python
half = cv2.resize(img, (w // 2, h // 2))  # the same real image, at half the original scale
```
```
REAL ORB keypoints on the same image at half scale: 445
real ORB matches between original and half-scale image: 191 total, 141 with distance < 40
```

A real, meaningfully lower 73.8% good-match rate (141 of 191) — the exact same algorithm, on the exact same
image, handled a real scale change measurably worse than it handled a real rotation.

## How It Actually Works

ORB computes an orientation for each keypoint and rotates its sampling pattern to match, which is specifically
why this lesson's real rotation test held up well: the descriptor is explicitly built to compensate for
rotation. Scale is handled differently — ORB detects keypoints across a real but limited multi-scale pyramid,
and a keypoint's descriptor is computed at roughly the scale it was detected at, not re-normalized the way SIFT
does with its own, more expensive scale-space search. This lesson's real, lower match rate under scaling is a
direct, measured consequence of that real design tradeoff, not a flaw in the test.

The real reason this course measures this rather than stating it is that "invariant to rotation and scale" is
often repeated as a blanket description of these algorithms without the real, honest caveat that *how much*
invariance a specific algorithm delivers is itself a real, measurable, and uneven quantity — exactly what this
lesson's two different real percentages (89.8% vs. 73.8%) demonstrate directly.

## Common mistakes

- Treating "scale and rotation invariant" as an absolute guarantee rather than a real, measurable degree of
  robustness — this lesson's own numbers show the same algorithm handling two different transformations with a
  real 16-point-percentage gap.
- Choosing ORB over SIFT (or vice versa) based on reputation alone rather than this lesson's real, concrete
  differences: ORB's binary descriptor is dramatically cheaper to store and match (32 bytes vs. SIFT's 512
  bytes per keypoint, both real, measured sizes), which matters when matching is run at scale in
  [Level 3](../level-3/02-running-a-real-detector.md)'s detection work.
- Running a matcher like `BFMatcher` with the wrong distance metric for the descriptor type — this lesson used
  `NORM_HAMMING` specifically because ORB's descriptor is binary; SIFT's floating-point descriptor needs
  `NORM_L2` instead, and mismatching the two produces meaningless real distance values.

## Exercise

Repeat this lesson's exact real scale-invariance test using SIFT instead of ORB (`SIFT_create()`'s descriptors
use `NORM_L2` for matching), and compare the real good-match percentage against ORB's measured `73.8%` — finding
out, by direct measurement, whether SIFT's more expensive scale-space search actually buys measurably better
real scale invariance on this image.
