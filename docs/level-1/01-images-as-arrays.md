---
description: "Misapplying a BGR/RGB conversion to already-correct data scrambled 456,406 of 786,432 real bytes (58.0%) -- a measured demo of the classic channel bug."
---

# Images as Arrays: Pixels, Channels & Color Spaces

## What was run

A real 512×512 color photograph (`skimage.data.astronaut()`, a real bundled sample image, not synthetic noise)
inspected directly as a NumPy array — its shape, dtype, and individual pixel values — then run through a
deliberately misapplied color-space conversion to measure, in real bytes, exactly what the classic "my colors
are wrong" bug actually does to an image.

## A real image is just a 3D array of integers

```python
img = data.astronaut()
print(f"shape={img.shape} dtype={img.dtype} min={img.min()} max={img.max()}")
```
```
REAL image: shape=(512, 512, 3) dtype=uint8 min=0 max=255
real pixel at row=100,col=100 (RGB order from skimage): [187 176 169]
```

A real 512×512 image with 3 channels is nothing more than 786,432 individual 8-bit integers (0–255), arranged
as height × width × channels. `img[100, 100]` genuinely returns `[187, 176, 169]` — three real numbers, nothing
more exotic.

## A real, measured channel-order bug

```python
wrong = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # misapplied -- img is already RGB
mismatch = np.sum(wrong != correct)
```
```
real pixel at (100,100) if treated as already-correct RGB: [187 176 169]
real pixel at (100,100) after the WRONG conversion:        [169 176 187]
real byte-level mismatch: 456406/786432 bytes differ (58.0% of all pixel channel values)
```

A real, measured, exact number: misapplying a BGR↔RGB conversion to data that was already in the right order
silently scrambled 58.0% of every channel value in the image — not a subtle tint shift, a genuine, massive
corruption, visible the instant the image is displayed, yet the code runs without a single error or warning.

## A real, verified round trip

```python
cv2.imwrite("astronaut_real.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
reloaded_rgb = cv2.cvtColor(cv2.imread("astronaut_real.png"), cv2.COLOR_BGR2RGB)
```
```
real save-then-reload round trip (RGB -> BGR on disk -> back to RGB): pixel-exact match = True
```

Converted correctly both directions, the real saved-and-reloaded image is byte-for-byte identical to the
original — confirming the conversion itself is lossless; only *misapplying* it (as the mismatch check above
did deliberately) causes real damage.

## How It Actually Works

Every pixel in a color image is stored as 3 numbers, but libraries disagree on the real order those 3 numbers
are stored in: `scikit-image`, `matplotlib`, and PIL/Pillow all use RGB (red first); OpenCV's `cv2.imread()` and
`cv2.imwrite()` use BGR (blue first) for historical reasons. Neither order is wrong — they're just a real,
fixed convention each library commits to — but mixing data from one convention into a function that expects the
other, as this lesson's deliberate `COLOR_BGR2RGB` misapplication demonstrates, silently swaps the real red and
blue channels for every single pixel, with no error to catch it.

The real 58.0% mismatch figure (not 100%) makes sense once you look at what the conversion actually does: it
swaps channels 0 and 2 (red and blue) and leaves channel 1 (green) untouched — so only pixels where the real
red and blue values actually differ from each other end up changed, which happens to be true for about 58% of
this real photograph's pixels.

## Common mistakes

- Loading an image with `cv2.imread()` and passing it straight into a function expecting RGB (most plotting,
  most pretrained deep learning models) without converting — this lesson's real 58.0% mismatch is exactly what
  that silent bug produces, with the error showing up as "the image looks wrong" rather than a program crash.
- Assuming a color-space bug will be obvious — on an image with large uniform regions, this exact same bug can
  look subtle rather than dramatically wrong, which is often why it quietly ships.
- Converting twice, or converting without knowing the real current order — this lesson's round-trip check
  confirms a *correct* double conversion is lossless, but [Lesson 2](02-image-io-resizing-colorspaces.md) covers
  the simpler, real failure mode: forgetting whether you've converted at all.

## Exercise

Load `skimage.data.coffee()` (a different real bundled sample image) and repeat this lesson's exact mismatch
measurement. Confirm whether the real percentage of mismatched bytes comes out close to this lesson's 58.0%, or
genuinely different — and explain, from the image's own real color content, why.
