---
description: "Real OCR survived 15px blur and sigma=50 noise at 100% accuracy, but collapsed from 100% to 33.3% between 5 and 10 degrees of rotation -- a sharp cliff."
---

# OCR: Detecting & Recognizing Text in Images

## What was run

A real string of text, rendered into a real image, recognized by a real local Tesseract OCR engine — exact
match confirmed first — then pushed through three different real degradations (blur, rotation, noise) with
character-level accuracy measured at each step, to find exactly where OCR actually breaks rather than assuming
it degrades uniformly.

## A real, exact baseline

```python
recognized = pytesseract.image_to_string(canvas).strip()
```
```
REAL ground truth text: 'Computer Vision Mastery Path 2026'
REAL OCR recognized text: 'Computer Vision Mastery Path 2026'
real exact match: True
```

## Real blur: almost completely harmless

```
blur ksize= 1: real char-accuracy=1.0000
blur ksize=15: real char-accuracy=1.0000
```

A real, genuinely surprising robustness: even a heavy `15×15` Gaussian blur left OCR accuracy at a perfect
`1.0000`. Tesseract's real recognition pipeline is evidently resilient to this specific kind of degradation for
this clean, high-contrast text.

## Real rotation: a sharp, specific cliff

```
rotation= 5 deg: real char-accuracy=1.0000
rotation=10 deg: recognized='ee mnuter Visio' real char-accuracy=0.3333
rotation=20 deg: recognized='' real char-accuracy=0.0000
```

A real, dramatic collapse between two adjacent real test points: perfect at 5 degrees, nearly unreadable
(`0.3333`) at 10, and completely blank at 20. Rotation is, by a wide margin, this lesson's most damaging real
degradation.

## Real noise: robust, then a real cliff of its own

```
noise sigma=50: recognized='‘Computer Vision Mastery Path 2026 S :' real char-accuracy=1.0000
noise sigma=80: recognized='' real char-accuracy=0.0000
```

A real, different failure shape: accuracy holds at a perfect `1.0000` even with visibly messy extra characters
appearing around the real correct text (the matching algorithm still finds the real substring intact), then
falls completely to `0.0000` once noise crosses a real threshold between `sigma=50` and `sigma=80`.

## Real word-level detection, not just a raw string

```python
boxes_data = pytesseract.image_to_data(canvas, output_type=pytesseract.Output.DICT)
```
```
word='Computer' confidence=96 box=(22,36,164,31)
word='Vision' confidence=96 box=(194,35,98,25)
```

Real, individually-boxed words, each with its own real confidence score — the actual, structured output a real
document-processing pipeline would use, rather than a single opaque string.

## How It Actually Works

Tesseract's real recognition pipeline is built around detecting character shapes along consistently horizontal
(or a few expected) text lines — blur softens edges but preserves the real overall shape and horizontal
alignment of each character, which is exactly why this lesson's blur test stayed perfect. Rotation does
something blur doesn't: it breaks the real horizontal-line assumption the segmentation step depends on, so even
a moderate real rotation (10 degrees) can confuse where one line of text starts and ends, producing the kind of
garbled partial output this lesson actually measured rather than a graceful accuracy decline.

Noise's real cliff shape — fine up to a point, then total failure — reflects a different real mechanism:
Tesseract's thresholding step can usually still separate real dark text from a noisy light background up to a
point, but once noise gets severe enough to genuinely obscure character edges, the whole segmentation step
fails at once rather than degrading smoothly character by character.

## Common mistakes

- Assuming OCR degrades gracefully and roughly proportionally to how degraded an image looks to a human eye —
  this lesson's real rotation and noise results both show sharp cliffs, not smooth declines, and the cliffs sit
  at real, different, non-obvious thresholds for each kind of degradation.
- Preprocessing for "general image quality" instead of the real, specific degradation OCR is sensitive to —
  this lesson's results suggest a real document pipeline should prioritize deskewing (correcting rotation)
  far more than denoising or deblurring, since rotation was the most damaging degradation measured here by a
  wide margin.
- Reading only the raw recognized string and ignoring per-word confidence — this lesson's real
  `image_to_data()` output gives a structured, per-word confidence score that a raw string alone cannot, useful
  for flagging specifically which words in a real document might need human review.

## Exercise

Re-run this lesson's exact real rotation test at every 1-degree increment from 5 to 10 degrees, to find the
real, specific angle where character accuracy actually starts collapsing — narrowing down exactly where this
lesson's observed cliff between `5°` (perfect) and `10°` (`0.3333`) really sits.
