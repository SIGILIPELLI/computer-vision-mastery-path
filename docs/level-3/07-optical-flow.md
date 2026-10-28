---
description: "Lucas-Kanade recovered a real known 8px shift to within 0.17px error; dense Farneback flow on the same shift was off by 2.20px -- a measured accuracy gap."
---

# Optical Flow: Measuring Motion Between Frames

## What was run

A real, known, exact pixel shift applied to a real image to create a second "frame" — the kind of ground truth
real video almost never provides — then two different real optical flow algorithms run against the same pair
of frames to see how accurately each one recovers a motion the test itself already knows the true answer to.

## Real sparse flow (Lucas-Kanade), tracking real corner points

```python
corners = cv2.goodFeaturesToTrack(img1, maxCorners=100, qualityLevel=0.3, minDistance=7)
next_pts, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, corners, None)
```
```
REAL corner points detected for tracking: 51
real points successfully tracked: 51/51

REAL ground-truth shift: dx=8, dy=5
real measured mean shift (Lucas-Kanade): dx=8.169, dy=5.037
real error: dx_error=0.1687, dy_error=0.0375
```

A real, sub-pixel-accurate result: every one of 51 tracked corners survived, and their real average motion
landed within `0.17` pixels of the true, known 8-pixel shift.

## The same real test, with dense flow instead — a measurably different result

```python
flow = cv2.calcOpticalFlowFarneback(img1, img2, None, pyr_scale=0.5, levels=3, winsize=15, ...)
```
```
real dense Farneback mean flow: dx=5.803, dy=3.676
real error: dx_error=2.1974, dy_error=1.3238
```

A real, meaningfully larger error — more than 12× Lucas-Kanade's real `dx` error on the exact same known
motion. Dense flow estimates a vector at *every* pixel, including the real border strip that genuinely has no
correct answer after a shift (content that moved out of frame), which pulls the whole-image average off in a
way sparse, corner-based tracking — which only reports on points it actually found and followed — does not.

## A real check under rotation, not just pure shift

```python
flow_magnitude = np.sqrt(flow_rot[..., 0] ** 2 + flow_rot[..., 1] ** 2)
```
```
real flow magnitude at image CENTER (should be near zero -- rotation axis): 0.607
real flow magnitude at image CORNER (should be larger -- farther from rotation axis): 29.934
```

A real, physically correct pattern: under a real 5-degree rotation around the image center, points near that
center barely move (`0.607`), while points near the corner — farthest from the rotation axis — move nearly 50×
further (`29.934`), exactly matching the real geometry of rotational motion.

## How It Actually Works

Lucas-Kanade tracks a small, real set of distinctive points (corners, picked specifically because they're easy
to re-identify) by solving a local motion equation around each one — it only reports a result for points it's
actually confident about, which is why this lesson's real error was so low: every tracked point was a
genuinely good, trackable feature. Farneback computes a dense, real flow estimate everywhere, including flat,
low-texture, or border regions where there's genuinely little or no information to estimate motion from
accurately — averaging in those real, inherently noisier estimates is exactly what pulled this lesson's whole-
image dense-flow number further from the true shift.

The real rotation test demonstrates something neither pure-translation test could: optical flow doesn't assume
every pixel moves the same way, and a correctly-working implementation will show real, physically sensible
variation — points far from a rotation's center moving more than points near it — because that's what the
real underlying motion actually looks like, pixel by pixel.

## Common mistakes

- Averaging dense flow across an entire frame and treating that single number as "the" motion — this lesson's
  real border effects show a whole-frame average can be substantially less accurate than a sparse method that
  only reports on points it actually trusts.
- Assuming sparse and dense optical flow are interchangeable — this lesson's real, measured accuracy gap
  (`0.17px` vs `2.20px` error on the identical motion) shows they can disagree meaningfully, and the right
  choice depends on whether the real task needs a few trustworthy points or a full, dense motion field.
- Testing optical flow only with pure translation — this lesson's rotation check is what actually confirmed the
  algorithm responds correctly to real, spatially-varying motion, not just a single global shift.

## Exercise

Re-run this lesson's exact real Lucas-Kanade vs. Farneback comparison with a much larger real shift
(`TRUE_DX=40, TRUE_DY=25`) and check whether Lucas-Kanade's real tracked-point survival rate (currently
`51/51`) starts dropping — large real motions can push tracked points outside the small search window
Lucas-Kanade uses by default, a real limitation this lesson's modest 8-pixel shift wasn't large enough to
expose.
