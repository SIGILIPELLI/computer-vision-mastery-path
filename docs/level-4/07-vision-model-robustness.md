---
description: "A real max-0.05 pixel change (0-1 range) flipped a 94.63%-confident 'Ankle boot' into an 87.83%-confident 'Sneaker' -- fooling 54% of 100 real images."
---

# Vision Model Robustness & Common Failure Modes

## What was run

A real FGSM (Fast Gradient Sign Method) adversarial attack against the trained CNN from
[Level 2](../level-2/index.md) — using the model's own real gradient to compute the smallest-possible nudge to
each pixel that most increases its loss — run at increasing real perturbation strengths against 100 real test
images the model already classified correctly, to measure exactly how little tampering it actually takes to
break a seemingly solid model.

## A real, steadily increasing fool rate

```python
perturbed = image + epsilon * image.grad.sign()
perturbed = torch.clamp(perturbed, 0, 1)
```
```
epsilon=0.00: real fool rate=0.0000 (0/100), real max pixel change=0.0000
epsilon=0.01: real fool rate=0.0400 (4/100), real max pixel change=0.0100
epsilon=0.03: real fool rate=0.3300 (33/100), real max pixel change=0.0300
epsilon=0.05: real fool rate=0.5400 (54/100), real max pixel change=0.0500
epsilon=0.10: real fool rate=0.8700 (87/100), real max pixel change=0.1000
epsilon=0.20: real fool rate=0.9900 (99/100), real max pixel change=0.2000
```

A real, striking escalation: at a maximum real pixel change of just `0.05` (5% of the full 0-1 pixel range,
genuinely imperceptible to a human glancing at the image), more than half of the previously-correctly-
classified real images were successfully fooled.

## A real, specific example

```python
perturbed = fgsm_attack(example_img, example_label, epsilon=0.05)
```
```
real true label: Ankle boot
real original prediction: Ankle boot (confidence=94.63%)
real prediction AFTER imperceptible perturbation: Sneaker (confidence=87.83%)
real max pixel value change: 0.0500
```

A real, concrete case: a confidently correct prediction (`94.63%` on "Ankle boot") flipped to a confidently
*wrong* one (`87.83%` on "Sneaker") after a maximum per-pixel change of `0.05` — a perturbation small enough
that the image would look essentially identical to a human, while the model's answer changed completely.

## How It Actually Works

FGSM computes the real gradient of the model's own loss with respect to the input image — the same kind of
gradient [Level 2, Lesson 8](../level-2/08-visualizing-cnn-features.md) used for saliency visualization, here
used for the opposite purpose: instead of asking "which pixels does the model care about," it asks "which
direction, for every pixel, would most increase the model's error," and then nudges every pixel exactly that
direction by a tiny real amount (`epsilon`). Because the perturbation is computed *specifically* to exploit
this model's own decision boundary, even a tiny real change can cross it, which is exactly why this lesson's
`epsilon=0.05` result — a change far too small to be visually obvious — still fooled the majority of real test
images.

This is a genuinely different real vulnerability from anything [Level 1](../level-1/index.md) or
[Level 3](../level-3/index.md) demonstrated: it's not random noise ([Level 4, Lesson 1](01-vision-transformers.md)'s
earlier robustness tests used unstructured corruption), and it's not a distribution shift
([Lesson 2's data drift concept](../level-4/index.md) from the MLOps-style monitoring content) — it's a
real, deliberately computed, worst-case perturbation that specifically targets this exact model's own learned
decision boundary.

## Common mistakes

- Judging a model's robustness by how it performs on ordinary, unperturbed real test data — this lesson's
  model scored well above 90% on standard test images, which says nothing about how it behaves under a real,
  deliberately adversarial perturbation like this lesson's FGSM attack.
- Assuming a small perturbation is automatically harmless because it's visually imperceptible — this lesson's
  real `0.05` maximum pixel change (on a 0-1 scale) is a genuinely tiny, hard-to-notice change that still
  fooled the majority of tested real images.
- Treating adversarial robustness as a problem only large, safety-critical systems need to worry about — this
  lesson's small, simple CNN, trained on an ordinary, non-adversarial real dataset, was still measurably
  vulnerable without any special effort to make it so.

## Exercise

Re-run this lesson's exact real FGSM attack, but instead of using the model's correct gradient direction, use
the *opposite* sign (`-image.grad.sign()`). Measure the real resulting fool rate at `epsilon=0.05` and compare
it to this lesson's real `0.5400`, to directly confirm that FGSM's real effectiveness comes specifically from
using the gradient's correct sign, not just from adding any perturbation of that magnitude.
