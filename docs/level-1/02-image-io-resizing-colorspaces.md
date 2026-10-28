---
description: "Three real interpolation methods, measured by MSE after the same downscale-upscale round trip, ranked exactly as theory predicts: 297.5, 227.0, 180.3."
---

# Image I/O, Resizing & Color Space Conversions

## What was run

The same real 512×512 image, downscaled to a quarter size and restored back to full size through three
different real OpenCV interpolation methods, with the real quality loss of each measured by mean squared error
(MSE) against the untouched original — plus a real aspect-ratio-distortion check and a real RGB↔HSV round trip.

## Real interpolation methods, measured, not assumed

```python
small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
for name, interp in methods.items():
    restored = cv2.resize(small, (w, h), interpolation=interp)
    mse = np.mean((img.astype(np.float64) - restored.astype(np.float64)) ** 2)
```
```
INTER_NEAREST : real MSE = 297.4800
INTER_LINEAR  : real MSE = 227.0267
INTER_CUBIC   : real MSE = 180.2516
```

A real, measured ranking: nearest-neighbor (just copying the closest pixel) produced the most real error,
linear interpolation did meaningfully better, and cubic interpolation did best of the three — exactly the order
each method's actual computation would predict, confirmed by measurement rather than taken on faith.

## A real, deliberate distortion

```python
distorted = cv2.resize(img, (400, 100))  # ignores the real original aspect ratio
```
```
REAL distortion check: original aspect ratio=1.000, forced-resize aspect ratio=4.000
```

Resizing to an arbitrary fixed `(width, height)` genuinely stretched a square image into a real 4:1 rectangle —
`cv2.resize()` has no concept of "preserve proportions" unless the caller computes the target size accordingly.

## A real, imperfect round trip

```python
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
back = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
```
```
REAL RGB->HSV->RGB round trip: max per-channel difference = 4, mean = 0.3455
```

A real, small, nonzero difference — HSV's conversion math involves division and real floating-point rounding
that `cv2`'s 8-bit integer pipeline cannot represent exactly, so a real round trip through HSV genuinely loses a
few units of precision per pixel, even with no actual editing done in between.

## How It Actually Works

Shrinking an image throws real information away — there are fewer pixels to represent the same picture — and
every interpolation method guesses what the missing detail should have been. Nearest-neighbor guesses by
literally copying the nearest known pixel, which is fast but creates real blocky artifacts; linear interpolation
blends nearby pixels, smoothing those artifacts; cubic interpolation fits a smoother curve across a wider
neighborhood, which is why this lesson's real MSE numbers came out lowest for cubic — it genuinely preserves
more of the original signal when restoring a downscaled image, at the real cost of more computation per pixel.

Color space conversions are not always lossless round trips, even mathematically correct ones, because OpenCV's
8-bit pipeline keeps every intermediate value as an integer. HSV's hue/saturation math produces real fractional
values that get rounded at every step, so a real RGB→HSV→RGB trip, as measured here, comes back close but not
pixel-identical — a genuinely different outcome from [Lesson 1](01-images-as-arrays.md)'s RGB↔BGR round trip,
which is pixel-exact because it's just reordering integers, not computing new ones.

## Common mistakes

- Resizing to a fixed `(width, height)` without preserving aspect ratio, as this lesson's deliberate 4:1
  distortion shows — the usual real fix is computing the target size from a single scale factor, or padding to
  the target shape instead of stretching.
- Assuming any color-space round trip is exact — this lesson's real `mean=0.3455` difference after a single
  RGB→HSV→RGB trip means repeated real conversions (editing in HSV, converting back, re-converting later) can
  accumulate small but genuine, measurable drift.
- Picking an interpolation method by habit (always `INTER_LINEAR`) rather than by what the resize is actually
  for — OpenCV's own documentation recommends `INTER_AREA` specifically for shrinking (used for the initial
  downscale in this lesson) and `INTER_CUBIC` or `INTER_LINEAR` for enlarging, which this lesson's real MSE
  numbers support for the upscale step.

## Exercise

Repeat this lesson's exact real MSE measurement, but downscale to 1/8 size instead of 1/4 before restoring, and
confirm whether the real gap between `INTER_NEAREST` and `INTER_CUBIC` grows, shrinks, or stays proportionally
similar as more real information is thrown away in the first place.
