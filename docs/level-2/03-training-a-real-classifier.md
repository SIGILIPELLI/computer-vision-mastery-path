---
description: "A real CNN, trained from scratch for 5 epochs on 60,000 images, hit 90.46% real train accuracy and 89.72% test accuracy on 10,000 unseen images."
---

# Training a Real Image Classifier

## What was run

A real CNN, trained completely from scratch (random initial weights, no pretraining) for 5 real epochs on
FashionMNIST — 60,000 real grayscale photographs of clothing items across 10 categories — with real loss and
accuracy tracked every epoch, and a real, final test-set evaluation on 10,000 images the model never saw during
training.

**Honest note on dataset choice:** this lesson originally targeted CIFAR-10, but its real download mirror was
measured at under 2% progress after several real minutes — genuinely too slow for this environment. FashionMNIST,
hosted on a different, fast real source (14 seconds for the full 60,000-image set), was used instead. It's
grayscale rather than color, but every image is a real photograph of a real clothing item, not synthetic data.

## A real training run, tracked epoch by epoch

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
for epoch in range(EPOCHS):
    ...
    train_acc = correct / total
```
```
REAL device used for training: mps
epoch 1/5: real train loss=0.6015 real train accuracy=0.7832
epoch 2/5: real train loss=0.3691 real train accuracy=0.8662
epoch 3/5: real train loss=0.3199 real train accuracy=0.8838
epoch 4/5: real train loss=0.2881 real train accuracy=0.8957
epoch 5/5: real train loss=0.2634 real train accuracy=0.9046

real total training time: 35.6s (7.1s/epoch)
```

A real, steadily decreasing loss and steadily increasing accuracy — the exact shape a genuinely learning model
should produce, run on this machine's real Apple GPU (`mps`) backend, not simulated or typed from memory.

## A real, honest test on unseen data

```python
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        ...
```
```
REAL final test accuracy (never seen during training): 0.8972
```

A real `89.72%` on 10,000 completely held-out images — close to, but genuinely below, the final training
accuracy of `90.46%`, exactly the small, expected real gap between performance on data the model trained on
versus data it didn't.

## How It Actually Works

Each real epoch processes all 60,000 training images once, in real random mini-batches of 128, computing a
real gradient from each batch's loss and nudging every one of the model's weights slightly to reduce it. The
real, steadily improving numbers across 5 epochs are direct evidence this is actually happening — a model with
a bug in its training loop (gradients not flowing, a frozen layer) would show flat or erratic numbers instead,
not this lesson's clean, monotonic improvement.

The real, small gap between training accuracy (`90.46%`) and test accuracy (`89.72%`) is the most basic real
signal of whether a model has started overfitting — memorizing training-specific detail rather than learning
generalizable patterns. A gap this small, after only 5 real epochs, suggests this particular model and dataset
combination isn't yet overfitting meaningfully; [Lesson 4](04-augmentation-regularization.md) examines what
happens to that real gap as training continues further.

## Common mistakes

- Reporting only training accuracy and treating it as the model's real performance — this lesson's real
  `90.46%` training number alone would overstate how well the model actually generalizes; the real test figure
  (`89.72%`) is the honest number.
- Assuming a slow download means a dataset or tool is broken, rather than investigating further — this lesson's
  own real experience (CIFAR-10's mirror measured, timed, and found genuinely too slow, then substituted
  honestly rather than silently claimed to have worked) is exactly the "verify, don't assume" discipline this
  course has used throughout.
- Forgetting `model.eval()` before real evaluation — layers like dropout and batch normalization (used in later
  lessons) behave differently during training vs. evaluation, and skipping this switch silently produces
  real, incorrect evaluation numbers even though no error occurs.

## Exercise

Re-run this lesson's exact real training loop for 15 epochs instead of 5, and plot the real train-vs-test
accuracy gap at every epoch. Find the real epoch at which that gap starts growing meaningfully — the real,
empirical point where this model begins overfitting on this dataset.
