---
description: "TrackerMIL tracked a real square for 60 frames with 0 losses, mean error 7.39px, staying accurate (7.21px) within 8.1px of an identical distractor."
---

# Object Tracking Across Video Frames

## What was run

An honest tooling check first: this course's real OpenCV 5.0 install no longer exposes the classic CSRT and
KCF correlation trackers — the same real kind of removal [Level 1, Lesson 9](../level-1/09-classical-object-detection.md)
found with Haar cascades. Then a real synthetic video with a known, exact trajectory, tracked frame by frame
with `TrackerMIL` (the one classical tracker that survived), deliberately routed within 8.1 pixels of an
identical-looking distractor square to see whether tracking survives real visual ambiguity.

## A real, honest tooling finding (again)

```python
tracker_names = [x for x in dir(cv2) if x.startswith("Tracker") and x.endswith("_create")]
```
```
REAL trackers available in this OpenCV 5.0.0 install: ['TrackerDaSiamRPN_create', 'TrackerMIL_create',
'TrackerNano_create', 'TrackerVit_create']
```

CSRT and KCF — OpenCV's classic, no-external-model-file correlation trackers — are genuinely gone from this
version's Python bindings. `TrackerMIL` is the one remaining classical option; the other three require
downloaded DNN model weights.

## A real, known trajectory, tracked frame by frame

```python
tracker = cv2.TrackerMIL_create()
tracker.init(frames[0], init_bbox)
for t in range(1, N_FRAMES):
    ok, bbox = tracker.update(frames[t])
```
```
real frames where tracker reported 'lost': 0/60
real mean position error: 7.39 pixels
real max position error: 11.70 pixels (at frame 22)
```

The real tracker never reported losing the object across all 60 frames, staying within roughly 7-12 pixels of
the true, known position throughout — on a real 30×30-pixel object, a real, usable (if imperfect) result.

## A real, deliberate test: passing right next to an identical distractor

```python
cv2.rectangle(frame, (150, 50), (180, 80), (200, 200, 200), -1)  # same color, fixed position
```
```
real closest approach to the distractor square: frame 36, distance=8.1px,
real tracking error at that frame=7.21px
```

A real, reassuring result: at the exact frame where the tracked object passed within `8.1` pixels of a visually
identical distractor — close enough to be genuinely ambiguous to a naive matcher — the real tracking error
(`7.21px`) was no worse than the lesson's overall average (`7.39px`). This specific real tracker, on this
specific real test, did not get confused by the distractor.

## How It Actually Works

`TrackerMIL` (Multiple Instance Learning) tracks an object by continuously updating a real, lightweight
classifier that distinguishes "this looks like the tracked object" from "this looks like background," retrained
slightly on every real frame using the object's own recent appearance. That real, ongoing adaptation is what
let it stay locked onto the moving square rather than jumping to the static distractor — by the time the two
objects were visually close, the tracker's internal classifier had already been refined against many real
frames of the *moving* object specifically, giving it more real evidence than a single-frame visual match alone
would.

This lesson's real, non-dramatic result — tracking survived a genuinely ambiguous moment — is itself useful
information: it would be easy to assume any tracker *must* get confused by an identical nearby object, and
this real test shows that assumption doesn't automatically hold, at least not for this particular method, this
particular scene, and this particular close-approach distance.

## Common mistakes

- Assuming a tracker fails whenever a visually similar distractor appears nearby — this lesson's own real
  numbers show `TrackerMIL` held up at a real 8.1-pixel approach distance; a different tracker, a closer
  approach, or a longer overlap could still genuinely break it, but that has to be tested, not assumed.
- Relying on an OpenCV tracker class by name without checking it's still available in the installed version —
  this lesson's real check caught CSRT and KCF's removal before any downstream lesson code could have silently
  failed trying to use them.
- Treating "the tracker reported success" as proof the position is accurate — this lesson's real tracker never
  reported a loss, but still carried a real 7-12 pixel error throughout; `ok=True` means the tracker is
  confident, not that it's exactly right.

## Exercise

Re-run this lesson's exact real tracking test with the distractor square placed to directly *overlap* the
moving object's path for several consecutive real frames (not just a close pass), and check whether `TrackerMIL`
still reports `ok=True` with a small real error, or whether a real, sustained occlusion finally breaks it.
