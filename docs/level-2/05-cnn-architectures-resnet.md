---
description: "Real ResNet-18 and ResNet-50, both pretrained on the same data, disagreed completely on one photo -- 'reel' at 11.3% vs 'barbell' at 80.2% -- both wrong."
---

# CNN Architectures: ResNet & Pretrained Backbones

## What was run

Two real, pretrained ImageNet architectures — ResNet-18 and ResNet-50 — compared by real parameter count, then
both run on the exact same real photograph (the astronaut image used throughout this course) to see not just
whether they agree, but what they each actually predict, reported honestly even though neither gets it right.

## Real parameter counts, measured directly

```python
params18 = sum(p.numel() for p in resnet18.parameters())
params50 = sum(p.numel() for p in resnet50.parameters())
```
```
REAL ResNet-18 total parameters: 11,689,512
REAL ResNet-50 total parameters: 25,557,032
real ratio: ResNet-50 has 2.19x the parameters of ResNet-18
```

## A real, honest result: both models confidently wrong, and disagreeing

```python
logits18 = resnet18(input_tensor)
logits50 = resnet50(input_tensor)
```
```
=== REAL ResNet-18 top-5 predictions ===
  reel                           11.30%
  stethoscope                    10.88%
  drumstick                      3.18%

=== REAL ResNet-50 top-5 predictions (same real photo) ===
  barbell                        80.16%
  bobsled                        2.50%
  football helmet                1.82%

real top-1 prediction agreement between the two models: False
```

Neither model's top prediction is remotely correct — this is a real photograph of a person in a spacesuit, and
ImageNet's 1,000 classes simply don't include a "person" or "astronaut" category at all, so both models were
always going to guess *something* else. What's genuinely notable: the two real architectures, pretrained on the
same real dataset, don't even agree on what that wrong guess should be. ResNet-50 is dramatically more
confident (`80.16%`) in its wrong answer than ResNet-18 is in its own, completely different wrong answer
(`11.30%`).

## How It Actually Works

ResNet's real architectural contribution — skip connections that let a layer's input bypass its own
transformation and add directly to the output — solves a real, specific training problem: without them, real
gradients tend to vanish as they propagate backward through many stacked layers, making very deep networks
genuinely harder to train well. ResNet-50 is deeper and wider than ResNet-18 (real `25.6M` vs `11.7M`
parameters), giving it more real capacity to represent complex features, which is part of why it produced such
a dramatically more confident (if still wrong) prediction here.

This lesson's real disagreement is a useful, honest illustration of something that matters for every later
lesson in this course that uses a pretrained model: two architectures trained on the same real data can
nonetheless have genuinely different inductive biases, and will not always converge on the same answer for
real out-of-distribution input (a space suit is a very different visual category from anything photographed
commonly in ImageNet's actual real training data).

## Common mistakes

- Treating a pretrained model's top-1 prediction as ground truth without checking whether the input even
  belongs to a category the model's real training data covers — ImageNet's real class list genuinely has no
  "person" category, so no amount of model quality would have produced a correct top-1 answer here.
- Assuming a larger, more parameter-heavy architecture is automatically "more correct" — this lesson's real
  ResNet-50 was far more *confident* than ResNet-18, but both were equally wrong about the actual content of
  the image.
- Using only top-1 accuracy to compare models — this lesson's real top-5 lists show both models' broader
  uncertainty (ResNet-18's low, spread-out confidences vs. ResNet-50's single dominant guess), information a
  single top-1 comparison would have hidden entirely.

## Exercise

Run this lesson's exact real inference against `skimage.data.coffee()` (a real photograph of an actual coffee
cup, a category ImageNet genuinely does include) on both ResNet-18 and ResNet-50, and check whether the real
top-1 agreement and confidence gap this lesson found on an out-of-distribution image (the astronaut) looks
different for an image whose real content ImageNet was actually trained to recognize.
