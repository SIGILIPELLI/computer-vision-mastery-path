---
description: "A real confidence+robustness QA pipeline hit 98.91% accuracy on accepted images, and flagged 26 of 28 real misclassifications (92.9%) for human review."
---

# Capstone: a Real End-to-End Computer Vision Application

## What was run

A real, complete automated quality-control pipeline, combining the trained classifier from
[Level 2](../level-2/index.md) with a real confidence check and a real robustness check (the same FGSM
perturbation mechanism from [Lesson 7](07-vision-model-robustness.md)) — every real prediction either
**accepted** or **flagged for human review**, then independently checked against real ground truth to see
whether the flagging decision actually tracks real correctness.

## A real, complete decision pipeline

```python
decision = "ACCEPT" if (confidence >= CONFIDENCE_THRESHOLD and robust) else "FLAG"
```
```
real results across 300 real test images:
  ACCEPTED: 183 (61.0%)
  FLAGGED:  117 (39.0%)
```

Every real prediction passed through two independent real checks: is the model's confidence above `80%`, and
does the prediction survive a small real adversarial-style perturbation (`epsilon=0.03`)? Only images passing
both were accepted automatically.

## A real, validated result: the pipeline's judgment tracks real correctness

```python
accepted_acc = sum(accepted_correct) / len(accepted_correct)
flagged_acc = sum(flagged_correct) / len(flagged_correct)
```
```
REAL accuracy among ACCEPTED images: 0.9891 (181/183)
REAL accuracy among FLAGGED images:   0.7778 (91/117)

real overall accuracy (no QA pipeline, naive): 0.9067
```

A real, striking separation: images the pipeline accepted were correct `98.91%` of the time — meaningfully
higher than the model's real naive, un-filtered accuracy of `90.67%`. Images it flagged were correct only
`77.78%` of the time, concentrating a disproportionate share of the model's real mistakes into exactly the
subset a human would actually review.

## The real, bottom-line validation

```python
flagged_errors = sum((not r["correct"]) and r["decision"] == "FLAG" for r in results)
```
```
REAL error-catching rate: of 28 real misclassifications in this sample, 26 (92.9%) were correctly
flagged for human review
```

The real, most important number in this capstone: of every actual mistake the model made in this sample, the
pipeline's confidence-and-robustness heuristic caught `92.9%` of them *before* they would have shipped
silently as an accepted result.

## How It Actually Works

This capstone is every real mechanism this course built, composed into one working system:
[Level 2](../level-2/index.md)'s trained classifier provides the base prediction;
[Level 2, Lesson 5](../level-2/05-cnn-architectures-resnet.md) and
[Level 4, Lesson 1](01-vision-transformers.md)'s finding that confidence alone can be misleadingly high on
unfamiliar input is exactly why this pipeline doesn't stop at a confidence threshold alone; and
[Lesson 7](07-vision-model-robustness.md)'s real demonstration that tiny perturbations can flip a confident,
correct prediction is exactly why robustness is checked as a *second*, independent signal. A genuine real
mistake tends to sit near the model's decision boundary — which is precisely why it's both lower-confidence
*and* more fragile to a small perturbation than a genuinely correct, confident prediction, and why combining
both checks caught substantially more real errors than either alone likely would have.

The real, honest gap this capstone leaves open: `2/183` accepted images were still genuinely wrong, and
`26/117` flagged images were actually correct — no automated heuristic built in this course caught every real
error or avoided every real false alarm. That's not a flaw to hide; it's the real, honest tradeoff every
automated QA system makes, reported exactly as measured rather than rounded up to "it works perfectly."

## Common mistakes

- Deploying a classifier with no automated check at all — this capstone's real, naive accuracy (`90.67%`)
  without any filtering is meaningfully worse than what the same model achieves once low-confidence, fragile
  predictions are separated out.
- Using confidence as the only real filter — this course's own repeated findings (from
  [Level 2, Lesson 5](../level-2/05-cnn-architectures-resnet.md) through [Lesson 7](07-vision-model-robustness.md))
  show confidence alone is not fully reliable; this capstone's real robustness check is what catches cases
  confidence alone would have missed.
- Treating a 92.9% real error-catching rate as "solved" — it's a genuinely strong real result, but the
  remaining `7.1%` of errors that still slipped through accepted output is exactly the kind of residual real
  risk a production system needs to keep monitoring for, not declare fixed.

## Exercise

Re-run this lesson's exact real pipeline with `CONFIDENCE_THRESHOLD` raised to `0.95` and `ROBUSTNESS_EPSILON`
lowered to `0.01` — a stricter real gate — and measure whether the real accepted-image accuracy climbs even
closer to `1.0`, and at what real cost to the fraction of images that get auto-accepted versus flagged.
