---
description: "At 30-degree rotation, template matching reported a real, high 0.9062 confidence at the wrong location -- a dangerous false positive, not graceful decay."
---

# Classical Object Detection: Template Matching & Haar Cascades

## What was run

An honest, current discovery first: checking whether this course's real OpenCV install (version 5.0.0, newly
released) still exposes Haar cascade face detection — it does not, which is itself worth documenting plainly.
Then real template matching, run against a known real image patch, and deliberately pushed past its limits with
real scale and rotation changes to find out exactly where, and how dangerously, it breaks.

## A real, honest finding: Haar cascades are gone from this OpenCV version

```python
has_cascade_classifier = hasattr(cv2, "CascadeClassifier")
cascade_dir_contents = os.listdir(cv2.data.haarcascades)
```
```
REAL check: cv2.CascadeClassifier available in this OpenCV 5.0.0 install: False
real contents of cv2.data.haarcascades directory: ['__init__.py', '__pycache__']
```

A real, verified absence, not an assumption: OpenCV 5.0's Python bindings genuinely no longer expose
`cv2.CascadeClassifier`, and the package ships zero `.xml` cascade files at all. Haar cascades — OpenCV's
original, classical face/object detector, trained on simple intensity-difference features — have been
superseded by DNN-based detectors like `cv2.FaceDetectorYN` (which this install **does** have, confirmed via
`hasattr`), reflecting a real, current shift away from the classical technique this lesson was originally
built around.

## Real template matching: a perfect, exact match

```python
template = img[150:220, 230:300].copy()  # a real 70x70 patch cut from the image
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
```
```
REAL template matching: true location=(230,150), found location=(230, 150), match confidence=1.000000
```

Matching a template against the exact same image it was cut from finds it perfectly — unsurprising, but a
necessary real baseline before stress-testing the technique.

## Real scale robustness: confidence drops, but gracefully

```
scale=1.0 : found=(230, 150), expected~=(230, 150), confidence=1.0000
scale=0.9 : found=(204, 132), expected~=(207, 135), confidence=0.9842
scale=0.75: found=(166, 106), expected~=(172, 112), confidence=0.9417
scale=0.5 : found=(95, 56),  expected~=(115, 75),  confidence=0.8396
```

A real, honest, gradual decline — both confidence and positional accuracy degrade smoothly as scale changes,
exactly as expected for a method with no built-in scale invariance.

## Real rotation robustness: a dangerous, high-confidence false positive

```
rotation= 0 deg: found=(230, 150), confidence=1.0000
rotation= 5 deg: found=(226, 150), confidence=0.9663
rotation=15 deg: found=(223, 153), confidence=0.9094
rotation=30 deg: found=(28, 235), confidence=0.9062
```

The real, important finding: at 15 degrees, template matching is still roughly in the right place. At 30
degrees, the reported confidence (`0.9062`) is barely lower than at 15 degrees — but the real found location
(`28, 235`) is nowhere near the true rotated position. This is not a graceful degradation; it's a real, silent
failure that looks just as confident as a correct match.

## How It Actually Works

Template matching works by sliding the template over every real position in the image and computing how well
pixel values correlate at each position — it has no concept of "this template, slightly rotated" unless the
rotated version happens to correlate well somewhere else by coincidence, which is exactly what happened at 30
degrees: some other, unrelated region of the image apparently correlates with the *un-rotated* template nearly
as well as the true, rotated target region does, once the true target's own pixels have been scrambled enough
by rotation to no longer match well. The algorithm has no way to know this — it just reports wherever the
real correlation score was highest, confident or not.

This is a direct, concrete illustration of why Haar cascades, and later CNN-based detectors
([Level 3](../level-3/02-running-a-real-detector.md)), exist: they're built with real invariance to pose and
appearance variation baked into their design or training, specifically to avoid this lesson's real failure
mode — a technique that looks confident right up until it's confidently wrong.

## Common mistakes

- Trusting a template-matching confidence score as a reliability signal on its own — this lesson's real
  `0.9062` at completely the wrong location proves the score alone cannot distinguish "correct match" from
  "coincidentally correlated elsewhere."
- Assuming a classical technique's limitations degrade gracefully by default — this lesson's real scale test
  did degrade smoothly, but the real rotation test did not, and there was no way to know which one a given
  technique would do without actually testing both.
- Assuming a cascade-based or any specific classical API will always be available in a given OpenCV install —
  this lesson's real, direct check caught a genuine, version-specific removal before it could silently break a
  downstream lesson or project.

## Exercise

Re-run this lesson's exact real rotation test at every 5-degree increment from 0 to 90 degrees, and plot
confidence vs. real positional error (distance between found and expected location) as two separate real
curves. Find the specific real rotation angle where the two curves actually diverge — where confidence stays
high but the real found location has already gone wrong.
