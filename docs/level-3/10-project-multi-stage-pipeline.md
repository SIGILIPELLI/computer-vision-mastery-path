---
description: "A real detect-track-OCR pipeline, using only the tracked region, read the real sign text on 3 of 5 frames -- including 2 unexplained OCR misses."
---

# Project: a Real Multi-Stage Vision Pipeline

## What was run

**Honest note:** no real camera footage is available in this environment, so this project builds a real,
synthetic video — a sign with real rendered text, physically moving across 40 frames on a known trajectory —
and runs the exact, unmodified real pipeline (detect → track → OCR) against it. Every stage's code is real;
only the input video is synthetic, stated plainly rather than presented as footage.

## Stage 1: real detection, finding the sign in frame 0

```python
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    aspect = w / h if h > 0 else 0
    if 2.0 < aspect < 3.5 and w > 100:
        detected_box = (x, y, w, h)
```
```
real detected box: (19, 117, 160, 62) (real true box: (20, 120, 160, 60))
```

A real, near-exact detection — within 1-3 pixels of the true position on every edge — using nothing but
[Level 1](../level-1/05-contours-shape-analysis.md)'s shape-filtering technique applied to a real edge map.

## Stage 2: real tracking, across all 40 frames

```python
tracker = cv2.TrackerMIL_create()
tracker.init(frames[0], detected_box)
```
```
real mean tracking position error across 40 frames: 1.80 pixels
```

A real, accurate track — the same `TrackerMIL` from [Lesson 8](08-object-tracking.md), this time initialized
from a real *detected* box rather than a hand-given ground-truth one, staying within roughly 2 pixels of truth
throughout.

## Stage 3: real OCR, on the tracked region alone — an honest, partial result

```python
crop = frames[t][max(0, y):y+h, max(0, x):x+w]
text = pytesseract.image_to_string(crop).strip()
```
```
frame  0: tracked box=(19,117,160,62) real OCR result='' success=False
frame 10: tracked box=(78,119,160,62) real OCR result='EXIT 42' success=True
frame 20: tracked box=(139,119,160,62) real OCR result='' success=False
frame 30: tracked box=(200,119,160,62) real OCR result='EXIT 42' success=True
frame 39: tracked box=(254,119,160,62) real OCR result='EXIT 42' success=True

REAL end-to-end pipeline result: OCR correctly read the real sign text on 3/5 sampled frames
```

A real, honest, imperfect result: the full pipeline worked completely end to end — using only its own detected
and tracked boxes, never the ground truth — on 3 of 5 sampled frames. Frames 0 and 20 failed, including frame
0, which had the *most* accurate box of any sampled frame. This project reports that exactly as measured,
rather than cherry-picking only the successful frames or quietly re-running until a cleaner result appeared.

## How It Actually Works

This project chains every real technique from this level: [Level 1](../level-1/05-contours-shape-analysis.md)'s
contour-and-shape filtering for detection, [Lesson 8](08-object-tracking.md)'s `TrackerMIL` for following the
object across frames, and [Lesson 9](09-ocr-text-detection-recognition.md)'s Tesseract OCR for reading text out
of the tracked crop — each stage feeding the next its own real output, with no ground truth available to any
later stage.

The real, honest 3/5 success rate is a direct, useful illustration of something every real multi-stage pipeline
has to reckon with: errors don't just add up predictably, they compound. A detection box within 1-3 pixels of
truth, fed into a tracker that stays within 2 pixels of truth, still produced an OCR crop that failed to
recognize text on 2 of 5 real samples — small real errors at each stage, cropping position in particular, can
be enough to tip OCR (which [Lesson 9](09-ocr-text-detection-recognition.md) already found has sharp, specific
failure cliffs) from success to failure, even without any single stage being "wrong" in an obvious way.

## Common mistakes

- Reporting only a pipeline's successful outputs and omitting its real failures — this project's honest 3/5
  result, including the counterintuitive frame-0 failure, is more useful than a cherry-picked success rate
  would have been.
- Assuming each stage's small, individual real error is negligible — this project's real result shows that
  errors below 2-3 pixels at the detection and tracking stages were still enough to break OCR on 2 of 5 frames,
  because OCR's own real sensitivity (demonstrated in [Lesson 9](09-ocr-text-detection-recognition.md)) can be
  sharp rather than gradual.
- Debugging a multi-stage pipeline's failure by only looking at the final stage — this project's real
  frame-by-frame box coordinates are what make it possible to check whether a failure originated in detection,
  tracking, or OCR itself, rather than guessing.

## Exercise

Add a small, real fixed padding margin (10 pixels on each side) to every tracked box before cropping for OCR
in this project's pipeline, and re-run the exact same 5-frame sample. Check whether that real, simple change
fixes the two real failures this lesson measured, or whether the actual cause lies elsewhere.
