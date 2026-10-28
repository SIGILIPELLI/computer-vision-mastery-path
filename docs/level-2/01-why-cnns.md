---
description: "A PyTorch conv layer manually set to Sobel-X weights matched cv2.Sobel exactly (0.0 diff) at every interior pixel -- a CNN layer IS a classical filter."
---

# Why CNNs: Convolution, Pooling & Receptive Fields

## What was run

A real PyTorch `nn.Conv2d` layer, with its weights manually set to the exact same Sobel-X kernel from
[Level 1](../level-1/03-filtering-convolution.md), run against the same real image and compared directly to
`cv2.Sobel()` — plus real, measured checks of output-size formulas, shift-invariance under pooling, and
receptive-field growth across stacked layers.

## A real CNN layer computing exactly the same thing as a classical filter

```python
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
conv.weight[:] = torch.from_numpy(sobel_x_kernel).view(1, 1, 3, 3)
```
```
REAL CNN conv layer (manually set to Sobel-X weights) vs cv2.Sobel (full image): max diff=923.000000
REAL same comparison, interior pixels only (excluding the 1px border): max diff=0.000000
```

A real, exact match at every interior pixel (`0.000000`) — once the comparison excludes the image border. The
real `923.0` discrepancy on the full image is, investigated directly, the exact same kind of real border-padding
mismatch [Level 1, Lesson 3](../level-1/03-filtering-convolution.md) already found: PyTorch's `padding=1` pads
with zeros, while `cv2.Sobel()` reflects at the border — two real, different conventions, not a math error.
Interior pixels, unaffected by either convention, prove the two are computing the identical real operation.

## Real formulas, verified against real torch output

```python
def expected_output_size(in_size, kernel, stride, padding):
    return (in_size + 2 * padding - kernel) // stride + 1
```
```
kernel=3 stride=1 padding=0: formula predicts 30, real torch output is 30, match=True
kernel=3 stride=1 padding=1: formula predicts 32, real torch output is 32, match=True
kernel=3 stride=2 padding=0: formula predicts 15, real torch output is 15, match=True
kernel=5 stride=2 padding=2: formula predicts 16, real torch output is 16, match=True
```

Every real combination matched the textbook formula exactly.

## Real pooling and real receptive-field growth, both measured directly

```python
no_pool_diff = torch.mean(torch.abs(img_t - shifted_t)).item()
pool_diff = torch.mean(torch.abs(pool(img_t) - pool(shifted_t))).item()
```
```
mean abs difference WITHOUT pooling: 7.0621
mean abs difference WITH 2x2 max pooling: 5.7749
```
```python
center = out[0, 0, 15, 15]
center.backward()
affected = (x.grad[0, 0].abs() > 1e-8).sum().item()
```
```
1 stacked 3x3 conv layers: real measured receptive field area = 9 pixels (expected 3x3=9)
2 stacked 3x3 conv layers: real measured receptive field area = 25 pixels (expected 5x5=25)
3 stacked 3x3 conv layers: real measured receptive field area = 49 pixels (expected 7x7=49)
5 stacked 3x3 conv layers: real measured receptive field area = 121 pixels (expected 11x11=121)
```

A real, measurable reduction in sensitivity to a 1-pixel shift after pooling (`5.7749` vs `7.0621`), and real,
exact receptive-field areas — measured directly via gradient flow, not just computed from a formula — confirming
each stacked 3×3 layer genuinely grows the region of input that can influence one output unit by exactly 2
pixels per side.

## How It Actually Works

A convolutional layer is not a different *kind* of operation from the classical filters in
[Level 1](../level-1/03-filtering-convolution.md) — it's the exact same sliding-window weighted sum, with the
one real difference that a CNN's weights are *learned* rather than hand-designed, as this lesson's manually-set
Sobel weights prove directly by matching `cv2.Sobel()` pixel-for-pixel. Pooling reduces spatial resolution and,
as this lesson measured, reduces sensitivity to small real translations — not eliminating it, but genuinely
dampening it, which is exactly the kind of approximate shift-tolerance a classifier benefits from. Stacking
layers grows the receptive field — how much of the real input ultimately influences one output value —
linearly with each additional 3×3 layer, confirmed here by directly measuring gradient flow rather than trusting
the formula alone.

## Common mistakes

- Treating a CNN's internal filters as mysterious or fundamentally different from classical image filters —
  this lesson's real, exact interior-pixel match (manually set Sobel weights) proves a single conv layer *is*
  a classical filter; what's different is that, during training, the network discovers its own useful kernels
  instead of a human hand-designing them.
- Assuming max pooling makes a network fully shift-invariant — this lesson's real numbers show pooling reduces,
  but does not eliminate, sensitivity to a 1-pixel shift (`5.7749`, not `0`).
- Comparing two convolution implementations on a full image without checking border-pixel handling separately
  — this lesson's real investigation (same discipline as [Level 1, Lesson 3](../level-1/03-filtering-convolution.md))
  is what correctly distinguished a real border-convention difference from an actual computation error.

## Exercise

Manually set a second real `nn.Conv2d` layer's weights to the Laplacian kernel from
[Level 1, Lesson 3](../level-1/03-filtering-convolution.md), and verify it matches `cv2.Laplacian()` at every
interior pixel the same way this lesson verified Sobel — confirming the same real principle holds for a second,
different classical filter.
