---
description: "Regularization eliminated the real overfitting gap (+0.0311 -> -0.0076) -- but the plain model's real test accuracy was still higher: 0.9192 vs 0.9034."
---

# Data Augmentation & Regularization for Vision Models

## What was run

Two real, identically-seeded CNNs, trained for 15 real epochs each on FashionMNIST: one plain, one with real
augmentation (random flips and rotations) and real dropout (0.4) — with the real train/test accuracy gap
tracked every single epoch for both, to see not just whether regularization "works," but what it actually
trades off.

## A real, growing overfitting gap, without regularization

```python
plain_model = FashionCNN(dropout_p=0.0)
```
```
epoch  1/15: train=0.7863 test=0.8302 real train-test gap=-0.0439
epoch  5/15: train=0.9078 test=0.8977 real train-test gap=0.0101
epoch 10/15: train=0.9322 test=0.9119 real train-test gap=0.0203
epoch 15/15: train=0.9503 test=0.9192 real train-test gap=0.0311
```

A real, textbook overfitting signature: the gap starts negative (the model is still underfit early on) and
climbs steadily positive as training continues, ending at a real `+0.0311` — training accuracy pulling ahead
of test accuracy, the model increasingly memorizing training-specific detail.

## A real, qualitatively different pattern with augmentation + dropout

```python
reg_model = FashionCNN(dropout_p=0.4)  # trained on real flip+rotation augmented data
```
```
epoch  1/15: train=0.7151 test=0.8249 real train-test gap=-0.1098
epoch  5/15: train=0.8631 test=0.8816 real train-test gap=-0.0185
epoch 10/15: train=0.8831 test=0.8991 real train-test gap=-0.0160
epoch 15/15: train=0.8958 test=0.9034 real train-test gap=-0.0076
```

A real, completely different shape: the gap stays *negative* for all 15 epochs — test accuracy consistently
**higher** than train accuracy, the opposite of the plain model's pattern.

## The real, honest complication

```
plain:          train=0.9503 test=0.9192 gap=0.0311
augmented+drop: train=0.8958 test=0.9034 gap=-0.0076
```

Regularization did exactly what it's supposed to do to the gap — but the real, final test accuracy tells a
more complicated story: the plain model's real `0.9192` is actually *higher* than the regularized model's
`0.9034`. Reducing overfitting did not, in this specific real run, translate into a better real test result.

## How It Actually Works

The regularized model's negative gap isn't actually a sign it's "better" at generalizing by some abstract
measure — it's a real, direct, measurable consequence of dropout's mechanics: during training, dropout
randomly disables 40% of the real units in that layer, artificially capping the model's measured training
accuracy at a point the *same* network, with dropout disabled during evaluation, can beat. That's precisely
why `model.eval()` (which turns dropout off) consistently reports a higher real number than `model.train()`'s
reported accuracy throughout this run.

The real, more important finding is that the regularized model's final test accuracy (`0.9034`) didn't beat
the plain model's (`0.9192`) despite a much cleaner train/test gap. This is a genuine, real instance of the
bias-variance tradeoff: this particular combination of augmentation strength and dropout rate reduced the
model's capacity to fit *anything*, including real, genuinely useful patterns, not just training-set noise —
and on a dataset of 60,000 real images that wasn't yet overfitting badly in the first place (the plain model's
own gap, `0.0311`, is modest), that tradeoff didn't pay for itself within 15 real epochs.

## Common mistakes

- Treating a shrinking (or negative) train/test gap as automatic proof that regularization "worked" — this
  lesson's own real numbers show the regularized model achieved a cleaner gap while still losing on raw real
  test accuracy.
- Applying strong regularization (augmentation + dropout together) without first checking whether the real,
  unregularized model is actually overfitting badly enough to need it — this lesson's plain model's real gap of
  `0.0311` after 15 epochs is a comparatively mild overfit; a model badly overfitting (a real gap of `0.15` or
  more) would likely show regularization paying off much more clearly.
- Forgetting that dropout changes measured training accuracy's *meaning*, not just its value — comparing a
  dropout model's real training accuracy directly to a non-dropout model's, as a measure of "how well is it
  learning," is comparing two numbers computed under genuinely different conditions.

## Exercise

Re-run this lesson's exact real comparison with a much weaker real regularization setting (`dropout_p=0.1`, no
augmentation) and check whether the real final test accuracy this time beats the plain model's `0.9192` — to
find, empirically, whether a lighter real regularization touch helps where this lesson's stronger one did not.
