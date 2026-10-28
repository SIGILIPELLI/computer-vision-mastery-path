---
description: "ViT-B/16 (86.6M params, 7.41x ResNet-18) agreed with ResNet-18 on in-distribution input ('espresso') but guessed 'bobsled' at 71.61% on the same OOD image."
---

# Vision Transformers: Concepts & Real Inference

## What was run

A real, pretrained ViT-B/16 (Vision Transformer) compared directly against ResNet-18 — real parameter counts,
real patch embedding mechanics, and real inference on the same two images from
[Level 2, Lesson 5](../level-2/05-cnn-architectures-resnet.md): one out-of-distribution (the astronaut photo,
no "person" class in ImageNet) and one in-distribution (the coffee cup, "espresso" is a real class).

## Real architecture facts, measured directly

```python
vit_params = sum(p.numel() for p in vit.parameters())
n_patches = (224 // vit.patch_size) ** 2
```
```
REAL ViT-B/16 parameters: 86,567,656
real ResNet-18 parameters: 11,689,512
real ratio: ViT-B/16 has 7.41x the parameters of ResNet-18

real ViT patch size: 16x16 pixels
real number of patches for a 224x224 input: 14x14 = 196 patches
```

A real, concrete picture of how ViT actually processes an image: it's split into exactly `196` non-overlapping
16×16 pixel patches, each one treated as a single "token" — the same basic unit a language transformer treats
a word as.

## Real agreement on in-distribution input, real disagreement on out-of-distribution input

```python
vit_out = torch.softmax(vit(inp_vit), dim=1)
```
```
--- astronaut (out-of-distribution) ---
ViT-B/16:    'bobsled' confidence=71.61% (inference: 105.7ms)
ResNet-18:   'reel' confidence=11.30% (inference: 17.3ms)

--- coffee (in-distribution) ---
ViT-B/16:    'espresso' confidence=90.35% (inference: 75.0ms)
ResNet-18:   'espresso' confidence=85.17% (inference: 12.3ms)
```

On the real, in-distribution coffee image, both architectures agree completely — "espresso," with ViT
reporting even higher real confidence. On the real, out-of-distribution astronaut image, they disagree
completely, exactly as [Level 2, Lesson 5](../level-2/05-cnn-architectures-resnet.md) already found between two
different CNN architectures — and ViT's wrong guess (`bobsled`, `71.61%`) is *far* more confident than
ResNet-18's wrong guess (`reel`, `11.30%`).

## A real, measured latency cost

A real, roughly `6×` slower inference time than ResNet-18 on both images (`105.7ms` vs `17.3ms`; `75.0ms` vs
`12.3ms`) — a direct, measured consequence of ViT's real `7.41×` larger parameter count.

## How It Actually Works

A Vision Transformer processes an image as a real sequence of patch tokens, using self-attention to let every
patch exchange information with every other patch directly — a fundamentally different real mechanism from a
CNN's local, sliding convolution. That architectural difference is exactly why ViT's and ResNet-18's wrong
guesses on the same out-of-distribution image diverge so completely: each model's real training produced a
different internal notion of "what does this unfamiliar input resemble," and transformers and CNNs don't
necessarily arrive at the same wrong answer, even when neither has the right one available.

This lesson's real finding extends [Level 2, Lesson 5](../level-2/05-cnn-architectures-resnet.md)'s result one
step further: it's not just that two *different CNNs* disagree on out-of-distribution input — a completely
different architecture family disagrees too, and does so with its own, independently high (and wrong)
confidence. Confidence alone, across any of these three real models, is not a reliable signal that the
model's class list actually contains the right answer.

## Common mistakes

- Assuming a larger, more modern architecture (ViT) is inherently more "honest" about uncertainty on
  unfamiliar input than an older one (ResNet) — this lesson's real numbers show the opposite: ViT's wrong guess
  was far *more* confident, not less.
- Choosing ViT over a CNN by default for its architectural novelty, without weighing its real, measured latency
  cost — this lesson's real `6×` slower inference is a genuine, practical tradeoff against its real parameter
  count and typical accuracy advantages on large, diverse training sets.
- Treating "transformer vs. CNN" as a question with one universally correct answer — this lesson's real
  agreement on in-distribution input and real disagreement on out-of-distribution input shows the right choice
  depends on the real problem and the real data the model will actually see.

## Exercise

Re-run this lesson's exact real comparison using `vit_b_32` (a ViT variant with `32×32` patches instead of
`16×16`, producing far fewer real tokens) in place of `vit_b_16`. Measure whether the real inference time drops
meaningfully with fewer patches, and whether the real out-of-distribution confidence pattern this lesson found
changes with coarser patches.
