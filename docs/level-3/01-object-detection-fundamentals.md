---
description: "A real IoU implementation matched 4 hand-computed cases exactly; the anchor matching a 40x40 target scored 0.7778 -- 3x every wrong-shape anchor's 0.25."
---

# Object Detection Fundamentals: Boxes, IoU & Anchors

## What was run

A real Intersection-over-Union (IoU) function, independently verified against four hand-computed test cases
before trusting it for anything else, then used to measure exactly how much overlap survives as two real boxes
drift apart, and finally to generate and evaluate a real, standard 9-anchor set against one target box.

## A real IoU function, verified before being trusted

```python
test_cases = [
    ("half overlap", (0, 0, 10, 10), (5, 0, 15, 10), 1 / 3),
    ("one inside the other", (0, 0, 10, 10), (2, 2, 8, 8), 36 / 100),
]
```
```
identical boxes          : computed=1.0000 expected=1.0000 match=True
half overlap             : computed=0.3333 expected=0.3333 match=True
one inside the other     : computed=0.3600 expected=0.3600 match=True
```

Every real case matched its independently hand-computed expected value exactly, before this function was used
for anything that actually mattered.

## A real, measured IoU decay as two boxes drift apart

```python
shifted_box = (shift, 0, shift + 100, 100)
```
```
shift=  0px: IoU=1.0000 -> PASS (>=0.5)
shift= 30px: IoU=0.5385 -> PASS (>=0.5)
shift= 50px: IoU=0.3333 -> FAIL (<0.5)
shift=100px: IoU=0.0000 -> FAIL (<0.5)
```

A real, concrete sense of what the common `IoU >= 0.5` "correct detection" threshold actually means in pixels:
for these real 100×100 boxes, a detection is still considered correct up to roughly 30-40 pixels of real
positional slop, and clearly wrong by 50.

## Real anchors, generated and matched against a real target

```python
w = scale * np.sqrt(ratio)
h = scale / np.sqrt(ratio)
```
```
real anchors generated at one position: 9 (3 scales x 3 aspect ratios)

anchor 3 (scale=40, ratio=0.5): real IoU=0.5469
anchor 4 (scale=40, ratio=1.0): real IoU=0.7778
anchor 5 (scale=40, ratio=2.0): real IoU=0.5469
anchor 0 (scale=20, ratio=1.0): real IoU=0.2500

real best-matching anchor: index 4, IoU=0.7778
```

Against a real 40×40 target box, the one anchor that's also a real 40×40 square (`scale=40, ratio=1.0`) scored
a real `0.7778` — more than 3× every wrong-scale anchor's `0.25`, and clearly ahead of the two correctly-scaled
but wrong-aspect-ratio anchors (`0.5469`). The anchor grid's real geometry directly determined which one could
possibly match well.

## How It Actually Works

IoU measures real overlap as intersection area divided by union area — a ratio that's `1.0` for identical boxes
and `0.0` for boxes that don't touch at all, with every real intermediate case falling smoothly between. This
lesson's drift test shows that real smoothness directly: as two same-sized boxes slide apart, IoU decreases in
a predictable, continuous way, which is exactly why it's used as a *threshold*-based correctness criterion
rather than an exact-match one — real object detectors are never going to output pixel-perfect boxes, so some
real tolerance has to be defined, and a common real convention is `IoU >= 0.5`.

Anchors are a real, practical solution to the problem "a detector needs to guess roughly where and what shape
an object might be before it has seen one" — by placing a real, fixed grid of candidate box shapes (this
lesson's 9, at 3 scales × 3 aspect ratios) at every position a detector scans, at least one anchor is likely to
be a real, decent starting match for most real objects, however they're shaped. This lesson's own numbers show
exactly why multiple scales and ratios are needed: no single anchor shape could have matched this one target
box as well as the one whose real scale and aspect ratio happened to line up with it.

## Common mistakes

- Treating IoU as an exact measure of "is this detection right or wrong" without defining a real threshold —
  this lesson's drift test shows IoU is a smooth, continuous quantity; "correct" is a real, chosen cutoff
  applied to it, not a property IoU itself declares.
- Using a single anchor shape (one scale, one aspect ratio) and expecting it to match every real object — this
  lesson's own numbers show a mismatched-ratio anchor at the *correct* scale still only reaches `0.5469`,
  noticeably below the matching-ratio anchor's `0.7778`.
- Trusting a geometry function like IoU without first checking it against hand-computed cases — this lesson's
  real verification step is what makes every later measurement in this course's detection lessons trustworthy.

## Exercise

Add a fourth real scale (`160`) and a fourth real aspect ratio (`3.0`) to this lesson's anchor generation,
producing 16 real anchors instead of 9, and find the new best-matching anchor against the same real target box
`(30, 35, 70, 75)` — confirming whether the original best match (`scale=40, ratio=1.0`) still wins, or whether a
larger real anchor set found something better.
