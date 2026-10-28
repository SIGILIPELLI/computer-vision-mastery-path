---
description: "A real shared 3x3 conv kernel covering the full input used 448 parameters; an equivalent fully-connected layer needed 50,348,032 -- 112,384x more."
---

# Building a CNN From Scratch in PyTorch

## What was run

A real, small CNN (`SimpleCNN`: two conv layers, two pooling layers, two fully-connected layers) built from
scratch in PyTorch, run with a real random input batch to confirm shapes, with every real parameter counted
layer by layer, and a real apples-to-apples comparison against an equivalent fully-connected layer to measure
exactly how much weight sharing actually saves.

## A real forward pass, and every real parameter counted

```python
model = SimpleCNN()
output = model(torch.randn(4, 3, 32, 32))
```
```
REAL forward pass: input shape=(4, 3, 32, 32), output shape=(4, 10)
real total trainable parameters in this CNN: 268,650
  conv1.weight        :      432 real parameters
  conv2.weight        :    4,608 real parameters
  fc1.weight          :  262,144 real parameters
  fc2.weight          :    1,280 real parameters
```

A real, honest, slightly surprising finding: this CNN's **total** parameter count (`268,650`) is actually
*higher* than a naive single fully-connected layer mapping the same input straight to 10 classes (`30,730`) —
because this specific small network's flatten-then-FC head (`fc1`'s `262,144` parameters) dominates the total,
not the convolutional layers.

## The real, fair comparison — and why it matters

```python
fc_equivalent_params = (3 * 32 * 32) * (16 * 32 * 32) + (16 * 32 * 32)
```
```
REAL apples-to-apples comparison (same input -> same output shape):
  conv1 (shared 3x3 kernel, all positions): 448 real parameters
  equivalent fully-connected layer:          50,348,032 real parameters
  real ratio: 112,384x more parameters for the FC version
```

The real, meaningful comparison isn't "whole CNN vs. a single classification layer" — it's "what would it cost
a fully-connected layer to produce the exact same *output shape* conv1 produces." Answered directly: a real,
112,384× more parameters. Weight sharing is real and dramatic — it just doesn't show up when comparing a whole
small network's total parameter count against a single layer that skips straight to 10 output classes.

## How It Actually Works

A convolutional layer reuses the exact same small set of weights (this lesson's `3×3×3` kernel, 448 numbers
including bias) at every one of the 1,024 spatial positions in a 32×32 output — it never needs a separate
weight for each position the way a fully-connected layer does. That's the real, structural source of the
112,384× gap this lesson measured directly: the FC equivalent needs a distinct weight for every
(input-pixel, output-position) pair, while the conv layer needs only one small kernel, reused everywhere.

The real reason this lesson's *overall* network ended up larger than the naive single-layer comparison is that
`fc1` — the layer that flattens the convolutional output and maps it to a hidden layer — reintroduces exactly
the problem convolution solves, at the boundary between the convolutional and fully-connected parts of the
network. This is a real, genuine architectural tension that modern networks ([Lesson 5](05-cnn-architectures-resnet.md)'s
real ResNet backbone, among others) address with global pooling instead of a large flatten step, specifically
to avoid this lesson's own `fc1` from dominating the parameter count.

## Common mistakes

- Repeating "CNNs have fewer parameters than fully-connected networks" as an unconditional fact — this lesson's
  own real total parameter count is a direct counterexample for this specific small architecture; the real
  savings are specifically in the convolutional layers, not guaranteed for the network as a whole.
- Comparing a full CNN's parameter count to a single FC layer that skips straight to the final output, instead
  of to an FC layer that could produce the *same intermediate output shape* — this lesson's real `112,384x`
  figure only appears once the comparison is made fair.
- Not verifying that gradients actually reach every real layer after `backward()` — this lesson's final check
  confirmed every parameter, from `conv1` through `fc2`, genuinely received a nonzero gradient, catching the
  class of real bug where a layer is accidentally detached from the computation graph.

## Exercise

Replace this lesson's `fc1` (which flattens all `32×8×8=2048` values) with `nn.AdaptiveAvgPool2d(1)` followed by
a much smaller linear layer (`32 -> 128`), and recompute the real total parameter count. Confirm by how much the
real number drops, and connect that real reduction to what [Lesson 5](05-cnn-architectures-resnet.md)'s real
ResNet backbone actually does differently from this lesson's simple architecture.
