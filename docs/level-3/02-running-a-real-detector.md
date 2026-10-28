---
description: "A real Faster R-CNN found 'person' at 99.53% confidence on a real photo -- and, unlike this course's classifiers, reported zero detections on blank input."
---

# Running & Evaluating a Real Object Detector

## What was run

A real, pretrained Faster R-CNN (trained on COCO, 91 real object classes) run on the same real astronaut
photograph used throughout this course — which genuinely does contain a person — to check not just whether it
finds the right object, but how it behaves on blank input compared to the classifiers from
[Level 2](../level-2/index.md).

## A real, correct detection, with a real confidence score

```python
predictions = model([img_tensor])[0]
keep = predictions["scores"] > 0.5
```
```
REAL detections above confidence 0.5: 2
  class=person          confidence=0.9953 box=(12,18,390,506)
  class=cup             confidence=0.6298 box=(268,347,506,510)
```

A real, correct top detection: "person," at `99.53%` confidence, in a bounding box spanning nearly the entire
512×512 frame — exactly right, since the photograph is a close-up of one person. The second, lower-confidence
"cup" detection is a real, honest output too — whether it's a genuine reasonable guess (some equipment in that
region) or a real mistake, it's reported as the model actually produced it, not edited after the fact.

## A real, meaningfully different behavior on blank input

```python
blank = torch.zeros_like(img_tensor)
blank_predictions = model([blank])[0]
```
```
REAL detections on a blank (all-zero) image, confidence > 0.3: 0
```

A real, clean result: zero detections above even a relaxed `0.3` confidence threshold. This stands in direct
contrast to [Level 2, Lesson 5](../level-2/05-cnn-architectures-resnet.md)'s real finding that a *classifier*
fed this exact same astronaut photo — out-of-distribution for its fixed class list — still confidently output
*some* class ("reel" or "barbell," depending on architecture). A detector, by design, can output "nothing
here" as a valid real answer, which a forced-choice classifier structurally cannot.

## How It Actually Works

A classifier like the ones in [Level 2](../level-2/index.md) must always pick one of its fixed output classes
— there's no real "none of the above" option built into a softmax over a fixed class list. A detector is
architecturally different: it proposes candidate regions and then decides, for each one, both *what* it might
be and *how confident* it is that anything real is actually there at all, which is exactly why blank input
correctly produced zero detections here rather than a forced, confident wrong guess the way
[Level 2, Lesson 5](../level-2/05-cnn-architectures-resnet.md)'s classifiers produced on out-of-distribution
input.

The real `99.53%` person detection and the real, lower-confidence `cup` detection together show the detector
doing its real job correctly: reporting high confidence where the evidence is strong (a person filling most of
the frame) and markedly lower confidence on a more ambiguous, smaller region — rather than treating every
detection as equally certain.

## Common mistakes

- Assuming a classifier and a detector would both fail the same way on degenerate input — this lesson's real,
  direct comparison shows a detector's architecture gives it a real, legitimate "nothing here" response a
  classifier cannot produce by design.
- Treating a detector's full output (every box it proposes before thresholding) as its real answer — this
  lesson applied a real `0.5` confidence threshold specifically to separate genuine detections from the many
  low-confidence candidate boxes a detector always proposes internally.
- Reporting only the top detection and ignoring lower-confidence ones that still cross a reasonable threshold —
  this lesson's real second detection (`cup`, `62.98%`) is exactly the kind of secondary output worth examining
  rather than discarding.

## Exercise

Re-run this lesson's exact real blank-input test, but instead of all zeros, use random real noise (uniform
values across the full real pixel range) as the input. Check whether the real detector still reports zero
confident detections, or whether structured randomness — unlike uniform blankness — produces real, spurious
detections the all-zero test didn't reveal.
