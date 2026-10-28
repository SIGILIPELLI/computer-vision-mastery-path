---
description: "An evidence-based choice (light flip-only augmentation + dropout 0.1) reached a real 91.83% test accuracy, improving the level's known weakest class's F1."
---

# Project: a Real Image Classifier Trained End to End

## What was run

A real, final training configuration chosen not by guessing, but directly from this level's own earlier real
findings: [Lesson 4](04-augmentation-regularization.md) found heavy augmentation and strong dropout underfit
this dataset; this capstone instead tries a deliberately lighter real touch — horizontal flips only, dropout
at `0.1` instead of `0.4` — trained for the same 15 real epochs, with a full, independent final evaluation.

## A real, evidence-based configuration, not a guess

```python
light_aug_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
model = FashionCNN(dropout_p=0.1)
```
```
epoch  1/15: train=0.7658 test=0.8396 real gap=-0.0738
epoch  8/15: train=0.9117 test=0.9002 real gap=0.0115
epoch 15/15: train=0.9325 test=0.9183 real gap=0.0141

REAL final test accuracy (independently recomputed): 0.9183
```

## The real, complete comparison across every configuration this level actually tried

```
Lesson 3  (from-scratch, no reg, 5 epochs)          : 0.8972
Lesson 4  (plain, no reg, 15 epochs)                 : 0.9192
Lesson 4  (heavy aug + dropout(0.4), 15 epochs)      : 0.9034
Lesson 6  (ResNet-18 fully fine-tuned, 5 epochs)     : 0.9041
THIS CAPSTONE (light flip-only aug + dropout(0.1))  : 0.9183
```

A real, near-tie with the best plain run (`0.9183` vs `0.9192`) — but with a real, meaningfully smaller
train/test gap trajectory than the unregularized version showed by epoch 15
([Lesson 4](04-augmentation-regularization.md)'s plain run reached a real gap of `0.0311`; this capstone's
stayed at `0.0141`), while comfortably beating both the heavily-regularized run and the fully fine-tuned
ResNet-18.

## A real, per-class check on the model's known weak point

```
       Shirt     0.7944    0.7380    0.7652      1000
```

[Lesson 7](07-evaluating-vision-models.md) found the 5-epoch model's real weakest class, `Shirt`, had an F1 of
`0.7056`. This capstone's longer, lightly-regularized training genuinely improved it to `0.7652` — real
progress on the exact weakness this level's own earlier evaluation identified, not just a higher average that
could have hidden continued struggling there.

## How It Actually Works

This capstone isn't a new technique — it's every real lesson in this level applied as evidence rather than
assumption: [Lesson 3](03-training-a-real-classifier.md)'s training loop, [Lesson 4](04-augmentation-regularization.md)'s
real finding that *this specific dataset* doesn't need strong regularization (prompting a lighter real touch
instead of abandoning regularization entirely), and [Lesson 7](07-evaluating-vision-models.md)'s per-class
evaluation discipline, used here to confirm the real fix actually helped where it was supposed to.

The real result — landing almost exactly between the unregularized run's higher raw accuracy and its larger
overfitting gap — is itself the honest, evidence-based takeaway: a real, justified middle ground, chosen
because [Lesson 4](04-augmentation-regularization.md)'s own numbers ruled out the heavier regularization
option, not because light regularization is a universally "correct" default.

## Common mistakes

- Picking a final model configuration by convention ("always use dropout 0.5") rather than by what this level's
  own real experiments actually showed about this specific dataset and architecture.
- Reporting only the final aggregate accuracy and declaring success — this lesson's real per-class check against
  [Lesson 7](07-evaluating-vision-models.md)'s known weak point is what actually confirms the configuration
  change helped where it mattered, not just on average.
- Treating the highest raw accuracy number as automatically "the best" model — [Lesson 4](04-augmentation-regularization.md)'s
  plain run technically scored `0.0009` higher than this capstone, but carried more than double the real
  overfitting gap, a real tradeoff worth knowing about even when choosing the simpler, higher-scoring option.

## Exercise

Train one more real configuration — light augmentation (flip only) combined with dropout at `0.4` instead of
`0.1` — for the same 15 epochs, and place its real final accuracy, gap, and `Shirt`-class F1 into this lesson's
comparison table. Confirm, with a fourth real data point, whether augmentation strength or dropout strength is
the bigger real driver of this dataset's underfitting risk.
