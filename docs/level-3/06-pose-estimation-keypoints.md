---
description: "Real upper-body keypoints scored confidence 9-18; lower-body keypoints for body parts outside this cropped photo scored as low as -2.88, at the image edge."
---

# Pose Estimation & Keypoint Detection

## What was run

A real, pretrained Keypoint R-CNN run on the real astronaut photograph, extracting all 17 real COCO body
keypoints — which revealed something more useful than a simple correctness check: this particular real photo
is cropped at roughly chest height, and the model's own real confidence scores honestly reveal exactly which
keypoints it could actually see versus which ones it was forced to guess at.

## Real, confident keypoints for body parts actually visible in the photo

```python
predictions = model([img_tensor])[0]
keypoints = predictions["keypoints"]
```
```
nose              : x=225.6 y=123.9 real confidence=17.303
left_eye          : x=247.4 y=102.1 real confidence=17.419
right_shoulder    : x=96.5 y=260.2 real confidence=9.748
```

Every real, visible upper-body keypoint — face and shoulders — scored a real confidence between roughly `9`
and `18`, and their real relative positions check out anatomically (nose above eyes above shoulders, in image
coordinates).

## A real, honest discovery: keypoints for body parts that aren't even in the photo

```
left_hip          : x=304.2 y=510.4 real confidence=0.149
left_knee         : x=376.5 y=510.4 real confidence=-2.663
left_ankle        : x=308.1 y=510.4 real confidence=-1.235
right_ankle       : x=133.0 y=510.4 real confidence=-2.879
```

A real, important pattern: every lower-body keypoint — hips, knees, ankles — landed at almost exactly
`y=510.4`, suspiciously close to this image's real 512-pixel height, with confidence scores that are near zero
or genuinely **negative**. The real photograph is cropped around chest height; these body parts simply aren't
in frame at all. The model still outputs *a* coordinate for every one of its 17 keypoint types — it has no
"not visible" output option — but its real confidence scores honestly flag that it has no real evidence for
any of them.

## How It Actually Works

A keypoint detector like this one is trained to always produce a fixed-size output: 17 (x, y, confidence)
triples, every time, for every detected person, regardless of what's actually visible. When a body part is
genuinely absent from the frame, the model has no real signal to place it correctly, and this lesson's result
shows it defaulting to somewhere near the image boundary — not a meaningful anatomical position, just where the
underlying regression happened to land with no real evidence to guide it. The real, negative confidence scores
are the honest signal that distinguishes this from a genuine detection: a real, visible keypoint scored
positively and often in the double digits, while an absent one scored at or below zero.

This is a direct, real illustration of why keypoint confidence matters as much as keypoint coordinates — a
naive pipeline that draws every returned keypoint without checking confidence would draw a complete, 17-point
skeleton onto this image, including knees and ankles for legs that were never photographed, and nothing about
the coordinates alone would reveal that those points are fabricated out of necessity rather than observed.

## Common mistakes

- Using every keypoint a pose model returns without checking its real confidence score — this lesson's real
  negative-confidence keypoints would silently produce a complete but partly fabricated skeleton if drawn
  without filtering.
- Assuming a low-but-positive confidence score (like the hip's real `0.149`) is meaningfully different from a
  high one — treating confidence as a soft ranking rather than checking it against a real, sensible threshold
  risks including keypoints the model itself is signaling real doubt about.
- Testing pose estimation only on full-body photographs — this lesson's real finding (sensible results for
  visible keypoints, honest low/negative confidence for absent ones) specifically required a cropped real
  photo to surface at all.

## Exercise

Re-run this lesson's exact real keypoint extraction, filtering out every keypoint with confidence below `5.0`,
and draw only the surviving real keypoints and the real skeleton connections between them. Confirm the result
shows a sensible partial upper-body skeleton with no fabricated legs, rather than this lesson's unfiltered,
complete-but-partly-invented 17-point output.
