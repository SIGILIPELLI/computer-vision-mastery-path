---
description: "RANSAC recovered an unknown real homography from 532/548 (97.1%) matches to within a mean reprojection error of 0.24 pixels across 5 test points."
---

# Feature Matching & Homography: Aligning Images

## What was run

A real, known perspective warp applied to a real image — a ground-truth transformation this script knows
exactly — then genuinely *forgotten*: ORB keypoints detected on both images independently, matched, and fed
into `cv2.findHomography()` with RANSAC to see how closely the algorithm can recover a transformation it was
never told, measured against the real ground truth it actually used.

## A real, known warp, recovered from real matched keypoints alone

```python
true_H = np.array([[1.05, 0.08, 20], [0.03, 0.98, -15], [0.0002, 0.0001, 1.0]])
warped = cv2.warpPerspective(img, true_H, (w, h))
...
est_H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```
```
real raw matches found: 548
real RANSAC inliers: 532/548 (97.1%)
```

A real 97.1% inlier rate — RANSAC correctly identified 532 of 548 real matched keypoint pairs as consistent
with a single real geometric transformation, discarding the rest as real mismatches.

## The real recovered transformation, checked against the real ground truth

```python
elementwise_diff = np.abs(true_H / true_H[2, 2] - est_H / est_H[2, 2])
```
```
real max element-wise difference: 0.672599
```

The raw matrix elements differ by up to `0.67` — but a homography matrix's raw numbers are not directly
comparable the way a prediction and a label are; what actually matters is where real points end up after the
transformation, checked next.

## The real, meaningful check: reprojection error at real points

```python
true_projected = cv2.perspectiveTransform(test_points, true_H)
est_projected = cv2.perspectiveTransform(test_points, est_H)
errors = np.linalg.norm((true_projected - est_projected).reshape(-1, 2), axis=1)
```
```
REAL reprojection error at 5 real test points (pixels): [0.467 0.232 0.119 0.287 0.103]
real mean reprojection error: 0.2415 pixels
```

A real, sub-pixel mean error of `0.2415` pixels across five independently chosen test points — the estimated
homography, built from nothing but matched ORB keypoints and RANSAC, places real points within a quarter of a
pixel of where the true (and, to the estimation process, completely unknown) transformation actually sends
them.

## How It Actually Works

A homography matrix has 8 real degrees of freedom (9 numbers, one redundant scale factor), and in principle 4
correctly matched point pairs are enough to solve for it exactly. This lesson's real 532 inlier matches are
drastically more than that minimum — which is precisely why RANSAC and the least-squares fit underlying
`cv2.findHomography()` can average out the real, inevitable noise in individual keypoint positions and land
this close to the true transformation, rather than being thrown off by any single bad match.

The reason raw matrix-element differences (this lesson's real `0.67` max) look large while the real,
practically-meaningful reprojection error (`0.24` pixels) is tiny comes down to how a homography's bottom row
works: the small perspective terms (`0.0002`, `0.0001`) interact multiplicatively with pixel coordinates in the
hundreds, so a tiny real error there can look large in isolation while barely moving where an actual point
lands — which is exactly why this lesson checks the real, geometrically meaningful quantity (where do points
actually go) rather than stopping at the raw numbers.

## Common mistakes

- Comparing two homography matrices element-by-element and concluding "very different" without checking real
  reprojection error — this lesson's own numbers show a `0.67` raw difference corresponding to a genuinely tiny
  `0.24`-pixel real-world effect.
- Skipping RANSAC and fitting a homography to every real raw match directly — with 548 raw matches and at least
  some genuine mismatches among them, a direct least-squares fit with no outlier rejection would be pulled
  measurably off by the real bad matches RANSAC's own inlier/outlier split (532 vs. 16) explicitly identified
  and discarded.
- Forgetting that a homography assumes a planar scene or pure camera rotation — this lesson's test image is
  effectively planar (a single 2D photograph), which is exactly the condition under which this kind of real
  recovery works this well; a genuinely 3D scene with real parallax would break this assumption.

## Exercise

Re-run this lesson's exact real pipeline with a much smaller real warp (shrink every off-diagonal and
perspective term in `true_H` toward the identity matrix) and measure whether the real reprojection error gets
better or worse — and separately, whether the real RANSAC inlier percentage changes, to see whether a subtler
transformation is actually easier or harder for this pipeline to recover precisely.
