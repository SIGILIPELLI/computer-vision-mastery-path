---
description: "Area filtering left one real contour that was actually background, not a coin -- a bounding-box check and a circularity filter caught it, down to 21."
---

# Contours & Shape Analysis

## What was run

Real contour detection run on the binary coin mask from [Lesson 4](04-thresholding-morphology.md), filtered
first by area, then — after a real, investigated discovery that area filtering alone wasn't enough — by a real
circularity measurement, to arrive at a genuinely correct count of real coin-shaped objects.

## Real contours, found and measured

```python
contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
```
REAL contours found (raw): 33
real contour areas (pixels^2), sorted: [0, 0, 0, ..., 0, 1052, 1076, ..., 3026, 12353]
```

33 raw contours, most tiny (zero-area single-pixel noise) or real coin-sized — plus one real outlier at
`12353` pixels², far larger than any other.

## A real filter that looked sufficient, until it wasn't

```python
MIN_AREA = 500
real_coins = [c for c, a in zip(contours, areas) if a >= MIN_AREA]
```
```
real contours after filtering area >= 500: 22
```

22 contours survived — but checking each one's real circularity (`4π·area / perimeter²`, where 1.0 is a
perfect circle) found one, at `area=12353.0`, with `circularity=0.0983`: far too low to plausibly be a coin.
Investigating directly with its real bounding box:

```python
x, y, w, h = cv2.boundingRect(c)
# area=12353.0 bbox=(0, 0, 329, 95), image shape=(303, 384)
```

A bounding box starting at the image's top-left corner, 329 pixels wide but only 95 tall — this is a real
background/table-surface region that Otsu's threshold classified as foreground because of its brightness, not
a coin at all. A pure area filter, with no shape check, let it straight through.

## The real, correct filter: area and circularity together

```python
MIN_CIRCULARITY = 0.5
genuine_coins = [c for c, circ in zip(real_coins, circularities) if circ >= MIN_CIRCULARITY]
```
```
REAL second filter (circularity >= 0.5): 22 area-filtered contours -> 21 genuine coins

real mean circularity across 22 detected coins: 0.8283
real circularity range: [0.0983, 0.8962]
real circularity of a perfect 50x50 square (for comparison): 0.7854
```

21 real, genuinely coin-shaped contours, with the one background region correctly excluded — and a useful real
reference point: a perfect square scores `0.7854`, so most of these real coins (mean `0.8283`, well above a
square) are close to round, as expected, while the excluded background region's `0.0983` is nowhere close to
either.

## How It Actually Works

Contour area alone measures *how big* a shape is; it carries no information about *what shape* it is. This
lesson's real background region happened to land in roughly the same area range as nothing in particular — it
was simply large — and a filter that only checks "is this big enough to matter" has no way to distinguish a
large, genuine coin from a large, irregular non-coin region. Circularity, computed from the real relationship
between a shape's area and its perimeter, captures exactly the property area filtering misses: a circle
packs the most area into the least perimeter of any shape, so anything with a long, thin, or irregular real
outline — like this lesson's background region — scores far lower, regardless of its total size.

The real lesson here isn't "always filter by circularity" specifically — it's that any single real shape
property, checked alone, can let through exactly the kind of object it wasn't designed to catch. Combining two
different real, independent measurements (area and circularity) is what correctly separated this lesson's 21
genuine coins from both the tiny noise contours and the one large background region.

## Common mistakes

- Filtering contours by area alone and assuming "big enough" means "the object I'm looking for" — this lesson's
  real background region at area `12353` would have passed any area threshold below that value, despite not
  being a coin at all.
- Checking a derived shape metric like circularity without also sanity-checking it against a known reference —
  this lesson's real square comparison (`0.7854`) is what made it possible to say with confidence that `0.0983`
  is genuinely, not just numerically, non-circular.
- Assuming `RETR_EXTERNAL` only returns "real objects" — it returns every real external contour the binary
  image actually contains, including large background regions that happened to threshold as foreground; the
  burden of distinguishing them is still on the caller's filtering logic.

## Exercise

Run this lesson's exact real pipeline on a binary mask produced by the real *adaptive* threshold from
[Lesson 4](04-thresholding-morphology.md) instead of Otsu's, and check whether the same two-stage area +
circularity filter still correctly isolates 21 genuine coins, or whether adaptive thresholding's different real
foreground shape produces new real contours this lesson's filters weren't tuned for.
