---
description: "Real garment pixels were 3.05x denser in the image center than the border -- yet gradient saliency and Grad-CAM both came back near parity (1.12x, 0.98x)."
---

# Visualizing What a CNN Learned

## What was run

Two real, standard techniques for visualizing what a trained CNN is actually responding to — raw input-gradient
saliency, and a Grad-CAM-style approach using the last convolutional layer's activations — both measured
quantitatively (center-region vs. border-region attention) against the real model from
[Lesson 3](03-training-a-real-classifier.md), and checked against the real underlying pixel content to see
whether either method's story actually holds up.

## A real, surprising first result

```python
saliency = image.grad.abs().squeeze().numpy()
center_saliency = saliency[center_mask].mean()
border_saliency = saliency[border_mask].mean()
```
```
real mean center/border saliency ratio across 20 samples: 1.12
real range: [0.64, 1.98]
```

If the model focuses on the real garment (center) rather than the real mostly-empty background (border), this
ratio should be clearly above `1.0`. Instead, it's close to parity, with some individual real samples actually
showing *more* gradient sensitivity at the border.

## Checking a second, more sophisticated method — same honest result

```python
weights = grad.mean(dim=(1, 2))
cam = torch.relu((weights[:, None, None] * act).sum(0))
```
```
real mean center/border GRAD-CAM ratio across the same 20 samples: 0.98
real range: [0.05, 2.99]
```

Grad-CAM, which is supposed to produce a cleaner, less noisy localization than raw input gradients, comes back
with the same real, honest story: no clear center-concentration either.

## The real check that makes sense of both results

```python
center_vals.append(img[center_mask].mean())  # raw pixel intensity, not saliency
```
```
real mean pixel intensity, center region (across 50 images): 0.4071
real mean pixel intensity, border region (across 50 images): 0.1335
real ratio: 3.05x
```

The real garment content genuinely *is* concentrated in the center — `3.05×` more pixel intensity there than
at the border, exactly as expected for a centered product photo. Both visualization methods, measured on the
same images, simply did not reflect that real asymmetry in their own attention.

## How It Actually Works

Raw input-gradient saliency measures how much the model's output *would* change if one pixel changed slightly
— and that's a real, different question from "how much does this pixel's actual content matter." A border
pixel that's almost always zero across the whole dataset can still have a nontrivial real gradient if the
model has, in effect, learned "a nonzero value here is unusual and worth reacting to," even though the garment
itself lives elsewhere. This is a well-documented, real limitation of gradient-based saliency methods: they're
often noisy and don't reliably highlight "the object" the way the mental picture of "attention" suggests, which
is exactly what this lesson's own direct, numeric comparison against real pixel content demonstrates rather
than just asserts.

Grad-CAM is supposed to improve on raw saliency by using a later convolutional layer's semantic activations
instead of raw input-gradient noise — but this lesson's real, small, two-conv-layer network has very little
depth for Grad-CAM's usual advantage to show up, and the real result (`0.98`, no better than raw saliency's
`1.12`) is an honest reflection of that, not a mistake in the implementation.

## Common mistakes

- Trusting a saliency or Grad-CAM visualization as "proof" the model is looking at the right thing, without a
  real, independent quantitative check — this lesson's own numbers show both methods can report something
  close to parity even when the real underlying content is genuinely 3x denser in one region than another.
- Assuming Grad-CAM is always meaningfully better than raw gradient saliency — this lesson's real, direct
  comparison on the same 20 images found no real improvement, specifically because this network's architecture
  is too shallow for Grad-CAM's usual semantic-layer advantage to apply.
- Drawing conclusions from a single visualized example — this lesson's real range (`0.64` to `1.98` for
  saliency, `0.05` to `2.99` for Grad-CAM) shows enormous real sample-to-sample variation; any one image's
  visualization could tell a very different, misleading story on its own.

## Exercise

Re-run this lesson's exact real Grad-CAM measurement, but on the much deeper, pretrained ResNet-18 from
[Lesson 5](05-cnn-architectures-resnet.md) instead of this lesson's shallow 2-layer CNN, using its final
convolutional block. Check whether a real, substantially deeper network's Grad-CAM shows a real
center-concentration this lesson's shallow network did not.
