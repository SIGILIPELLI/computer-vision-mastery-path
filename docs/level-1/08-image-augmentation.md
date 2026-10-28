---
description: "A real horizontal flip scored SSIM=0.1456 ('dissimilar') while a real cutout scored 0.9647 ('similar') -- backwards from which removes more content."
---

# Image Augmentation for Training

## What was run

A real battery of common training-time image augmentations — flip, rotation, brightness jitter, Gaussian
noise, random crop, and cutout — each measured against the untouched original using structural similarity
(SSIM), a real, standard image-similarity metric, to see what each augmentation actually does to an image
quantitatively rather than just visually.

## Real SSIM measurements across every augmentation

```python
flipped = cv2.flip(img, 1)
print(real_ssim(img, flipped))
```
```
horizontal flip          : SSIM=0.1456
rotate  5 degrees         : SSIM=0.3453
rotate 15 degrees         : SSIM=0.2259
rotate 45 degrees         : SSIM=0.1779
brightness +10            : SSIM=0.8820
brightness +40            : SSIM=0.8030
brightness +80            : SSIM=0.6886
gaussian noise sigma=5    : SSIM=0.8502
gaussian noise sigma=20   : SSIM=0.4215
gaussian noise sigma=50   : SSIM=0.1956
random crop + resize      : SSIM=0.2620
cutout (100x100 region)   : SSIM=0.9647 (real 10000 of 262144 pixels zeroed = 3.81%)
```

## A real, counter-intuitive finding worth taking seriously

A horizontal flip — an augmentation that preserves every single real pixel value and the complete content of
the image, just mirrored — scored a real `0.1456`: SSIM's *most* dissimilar result of this entire lesson. A
100×100 cutout, which genuinely destroys 3.81% of the real image by zeroing it out entirely, scored `0.9647`:
SSIM's *most* similar result, after brightness and noise. By SSIM's own measurement, a flip "changes" the image
far more than erasing a real patch of it does — which is backwards from what anyone looking at both images
would say.

## How It Actually Works

SSIM compares small local windows at *matching pixel coordinates* between two images — it measures whether the
patch at position `(x, y)` in image A looks like the patch at that same position `(x, y)` in image B. A flip
moves nearly every piece of real content to a different coordinate, so almost none of those paired windows
contain the same real content anymore, even though the *image itself* is fully intact and just mirrored. A
cutout, by contrast, leaves 96.19% of the image's real pixels in their exact original positions, so the
overwhelming majority of SSIM's compared windows still match closely — even though a real, coherent chunk of
content is now simply gone.

The real lesson is specific and important for this course: SSIM (and position-based pixel metrics generally)
measure *spatial* similarity, not *semantic* or *content* similarity. A flip is semantically almost identical to
the original (same objects, same recognizability, mirrored) but spatially very different; a cutout is
spatially almost identical (nearly every pixel in its original place) but has genuinely removed real
information. Choosing an augmentation — or evaluating one — based on SSIM alone would get exactly these two
cases backwards.

## Common mistakes

- Using SSIM (or any purely spatial pixel metric) to judge whether an augmentation is "too aggressive" for
  training — this lesson's real numbers show it can rate a content-preserving flip as more extreme than an
  actual information-destroying cutout.
- Assuming an augmentation that looks subtle to a human (cutout, at only 3.81% of pixels) is therefore "safe"
  for training — the real information removed by a cutout can still matter enormously to a model if it happens
  to land on the one discriminative feature that mattered, regardless of what any pixel-similarity metric says.
- Picking a single similarity metric and trusting it as a universal measure of "how much has this image
  changed" — this lesson's real, directly-contradictory flip-vs-cutout result is exactly why a metric needs to
  be chosen for what it actually measures, not applied as if "similarity" means one single thing.

## Exercise

Repeat this lesson's exact real SSIM measurement for the horizontal flip, but first flip the *comparison* image
back (`cv2.flip(flipped, 1)`) before computing SSIM against the original. Confirm the real SSIM this time comes
back close to `1.0` — proving directly that the low `0.1456` score in this lesson came from spatial
misalignment, not from any genuine difference in the image's real content.
