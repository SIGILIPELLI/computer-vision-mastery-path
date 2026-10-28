---
description: "Fine-tuning all 11.2M real ResNet-18 params reached 90.41% test accuracy, barely beating a tiny from-scratch CNN (89.72%) at 2.75x the training time."
---

# Transfer Learning & Fine-Tuning a Pretrained Model

## What was run

The real, pretrained ResNet-18 from [Lesson 5](05-cnn-architectures-resnet.md), adapted for FashionMNIST's
grayscale input, trained two real different ways — with its backbone completely frozen (only the new final
layer trained) and fully fine-tuned (every layer trained) — both for the same 5 real epochs, compared directly
against each other and against [Lesson 3](03-training-a-real-classifier.md)'s tiny from-scratch CNN.

## Real frozen-backbone transfer learning: fast, but limited

```python
for name, param in model.named_parameters():
    if not name.startswith("fc."):
        param.requires_grad = False
```
```
=== REAL run: frozen backbone, only fc trained (5,130/11,175,370 trainable params) ===
epoch 5/5: real train accuracy=0.7289
real total time: 107.9s (21.6s/epoch)
REAL test accuracy: 0.7269
```

Training only `0.046%` of the model's real parameters reached a real `72.69%` test accuracy — a real, working
result, but clearly behind what's possible with this architecture.

## Real full fine-tuning: better, but at a real cost

```python
# every real parameter, including the pretrained backbone, left trainable
```
```
=== REAL run: full fine-tune, all layers trained (11,175,370/11,175,370 trainable params) ===
epoch 5/5: real train accuracy=0.9324
real total time: 297.3s (59.5s/epoch)
REAL test accuracy: 0.9041
```

A real, meaningfully better `90.41%` — but at a real `2.75×` increase in per-epoch training time over the
frozen version.

## The real, honest comparison that matters most

```
frozen backbone : test_acc=0.7269 time=107.9s
full fine-tune   : test_acc=0.9041 time=297.3s
(for reference, Lesson 3's from-scratch small CNN reached 0.8972 after 5 epochs)
```

A real, humbling result: fully fine-tuning an 11.2-million-parameter pretrained ResNet-18, at nearly 3× the
real training cost, beat [Lesson 3](03-training-a-real-classifier.md)'s tiny from-scratch CNN (around 50,000
parameters) by only `0.69` real percentage points — `90.41%` vs. `89.72%`.

## How It Actually Works

A frozen backbone only lets the model learn a new way to *combine* features ResNet-18 already learned from
real ImageNet photographs — and those real features, learned from natural photos, don't transfer perfectly to
grayscale clothing items, which is exactly why the frozen run plateaued around `73%` rather than climbing
further. Full fine-tuning lets every real layer adapt its features to this new, real domain, which is why it
reached a real, higher accuracy — at the real cost of updating (and computing gradients for) every one of its
11.2 million parameters, not just the final 5,130.

The real, more important finding is what this comparison says about *when* transfer learning is worth its real
cost: for a domain reasonably close to natural photographs, a frozen or lightly fine-tuned pretrained backbone
usually provides a large, real head start over training from scratch. FashionMNIST's real grayscale, heavily
simplified images are different enough from ImageNet's real natural photos that this course's own tiny,
from-scratch CNN — architected specifically for this exact real task — nearly matched a vastly larger
pretrained model's fine-tuned result, at a fraction of the real compute.

## Common mistakes

- Assuming transfer learning is always the right choice over training from scratch — this lesson's real
  numbers show a small, task-specific architecture can come within a real percentage point of a much larger
  fine-tuned pretrained model, for a fraction of the real training cost.
- Freezing a pretrained backbone without first checking how different the real target domain is from the
  original pretraining data — this lesson's real `72.69%` frozen result reflects a genuine domain gap between
  ImageNet photographs and FashionMNIST's simplified grayscale images.
- Comparing training approaches by final accuracy alone, ignoring real training time — this lesson's real
  `2.75×` time difference between frozen and fully fine-tuned runs is a genuine, practical cost that matters
  as much as the real accuracy gain it bought.

## Exercise

Re-run this lesson's exact real frozen-backbone experiment, but unfreeze only the last residual block
(`model.layer4`) in addition to the final `fc` layer, instead of the entire network. Measure the real resulting
test accuracy and training time, to find whether this middle ground captures most of full fine-tuning's real
accuracy gain at closer to the frozen version's real training cost.
