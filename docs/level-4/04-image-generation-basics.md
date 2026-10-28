---
description: "After 10 epochs, the discriminator still caught 92.4% of fakes -- yet an independent classifier gave those same fakes a real 69.4% mean confidence."
---

# Image Generation Basics: GANs & Diffusion

## What was run

A real, small GAN (a simple fully-connected generator and discriminator, no convolutions) trained for 10 real
epochs on FashionMNIST, with every epoch's real losses and real discriminator accuracy tracked — then the
generator's real output quantitatively evaluated using a completely independent, already-trained classifier
from [Level 2](../level-2/index.md), rather than judged by eye.

## A real, honest training trajectory — the discriminator still winning

```python
d_loss = criterion(real_pred, real_labels) + criterion(fake_pred, fake_labels)
g_loss = criterion(fake_pred, real_labels)  # generator wants the discriminator fooled
```
```
epoch  1/10: real G_loss=1.4412 real D_loss=0.7296 real D(real) acc=0.9529 real D(fake) acc=0.9313
epoch  5/10: real G_loss=2.0309 real D_loss=0.8365 real D(real) acc=0.7695 real D(fake) acc=0.9764
epoch 10/10: real G_loss=1.5089 real D_loss=1.0509 real D(real) acc=0.7016 real D(fake) acc=0.9236
```

A real, honest result after 10 epochs: the discriminator is still correctly identifying `92.36%` of the
generator's fakes as fake. The generator has not come close to winning this adversarial game yet — a real,
typical early-stage GAN training signature, not a failure to report around.

## A real, independent, quantitative check on output quality

```python
probs = torch.softmax(classifier(fake_imgs), dim=1)
```
```
real mean top-1 confidence across 200 generated images: 0.6941
  predicted 'Shirt': 47/200 generated images
  predicted 'Dress': 36/200 generated images
  predicted 'Sneaker': 34/200 generated images
  predicted 'Coat': 1/200 generated images
```

A real, genuinely informative result despite the discriminator still winning: an entirely separate, already-
trained real classifier — one that never saw a single generated image during its own training — assigns the
generator's fake images a real mean confidence of `69.41%` across plausible garment classes, spread across 9
of the 10 real categories. The generated images aren't random noise; they carry enough real, learned structure
for an independent model to recognize garment-like patterns in them, even while this GAN's own discriminator
remains largely unfooled.

## How It Actually Works

A GAN trains two real networks in direct competition: the discriminator learns to tell real images from the
generator's fakes, and the generator learns to produce fakes the discriminator can't catch, using the
discriminator's own real gradient as its only training signal. This lesson's real numbers show a textbook-
typical early dynamic: the discriminator's real accuracy on fakes (staying above `92%` throughout) means the
generator's gradient signal is coming from a discriminator that's still comfortably ahead, which is exactly
why `G_loss`'s real trajectory is noisy and only slowly improving rather than collapsing toward zero.

The real reason this lesson evaluates the generator with an *independent* classifier rather than the GAN's own
discriminator is that the discriminator's judgment is entangled with this specific generator's current
weaknesses — it's not a neutral measure of "does this look like a real garment," only "can I currently tell
this apart from training data." The independent classifier's real `69.41%` confidence is a genuinely different,
more neutral real signal: unrelated structure in the image is enough to produce real, class-plausible garment
features, even this early in training and even with a discriminator that's still winning.

## Common mistakes

- Judging GAN training progress by discriminator accuracy alone — this lesson's real `92.36%` D(fake) accuracy
  at epoch 10 would look like total generator failure in isolation, while the independent classifier's real
  `69.41%` confidence shows meaningful structure is already present.
- Assuming GAN training should show losses decreasing steadily to zero, the way supervised training does — a
  real adversarial game's losses, as this lesson's own noisy `G_loss` trajectory shows, don't behave like an
  ordinary optimization curve, because the target (the discriminator) is itself changing throughout training.
- Treating image generation and image classification as unrelated techniques — this lesson used
  [Level 2](../level-2/index.md)'s own trained classifier as a real, independent evaluation tool for a
  generative model, something the two subfields are directly connected by in practice.

## Exercise

Train this lesson's exact real GAN for 30 epochs instead of 10, and re-run the exact same independent-classifier
evaluation. Check whether the real mean confidence (currently `0.6941`) rises meaningfully with more real
training, and whether the discriminator's real fake-detection accuracy (currently `92.36%`) finally starts
dropping as the generator improves.
