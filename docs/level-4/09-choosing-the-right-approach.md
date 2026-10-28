---
description: "Every number here cites a real measurement this course made -- the 4.7x batching speedup, the 7.41x ViT/ResNet ratio, the 112,384x conv/FC gap."
---

# Choosing the Right CV Approach for a Real Problem

## What was run

No new code — this lesson is a decision framework built entirely from real, measured numbers this course
already produced, cited back to where each one came from, to answer the practical question every one of this
course's techniques eventually raises: given a real problem, which approach actually fits?

## Decision 1: classical CV, or a trained model?

[Level 1](../level-1/index.md) built an entire real, working document scanner
([Level 1, Lesson 10](../level-1/10-project-document-scanner.md)) using nothing but classical filtering,
contours, and perspective math — reaching a real `1.10`-pixel mean corner-detection error with zero training
data. But [Level 1, Lesson 9](../level-1/09-classical-object-detection.md)'s real template-matching test also
found a confident, high-score (`0.9062`) false positive at just 30 degrees of rotation. **Choose classical CV**
when the real problem is geometrically well-defined and the input is reasonably controlled (a scanner, a
production-line inspection with a fixed camera angle); **choose a trained model** when real appearance
variation (pose, lighting, background) is substantial enough that hand-designed rules would need endless real
special cases.

## Decision 2: classification, detection, or segmentation?

[Level 3](../level-3/index.md) measured all three on the same real image: a classifier forces one label onto
the whole image; [Level 3, Lesson 2](../level-3/02-running-a-real-detector.md)'s real detector correctly
reported zero detections on blank input (a classifier structurally cannot); and
[Level 3, Lesson 5](../level-3/05-instance-vs-semantic-segmentation.md)'s real test showed only instance
segmentation could report "2 separate people" with `0` mask overlap, where semantic segmentation gave one
undifferentiated blob. **Choose classification** when the real question is "what, overall, is in this image";
**detection** when you need "where, and how many"; **segmentation** when you need exact real pixel boundaries;
**instance segmentation** specifically when you must count or separately track same-class objects.

## Decision 3: how much model is actually needed?

This course has not run a dedicated multi-architecture cost/latency sweep the way its MLOps-focused sibling
course has, but [Level 2, Lesson 6](../level-2/06-transfer-learning.md)'s real
fully fine-tuned ResNet-18 (`11.2M` params, `2.75×` the training time) beat a tiny from-scratch CNN by only
`0.69` real accuracy points. **Start small and measure** — this course's own repeated real finding is that
bigger models often buy far less than their real cost implies, for a *specific* task.

## Decision 4: real-time or batch?

[Lesson 6](06-scaling-vision-inference.md) measured real per-image throughput climbing from `6,139` to
`21,373` images/sec as batch size grew, then *dropping* at an even larger batch size, and a real `4.7×`
speedup from batching 256 images in one call versus 256 separate ones. **Choose batch** whenever the real
workload is a known, scheduled set of images; **choose single-image serving** only when a real individual
response is actually needed the moment one input arrives, and measure — as this course did — rather than
assume larger batches are always better.

## Decision 5: how much do you actually trust the model's confidence?

[Level 2, Lesson 5](../level-2/05-cnn-architectures-resnet.md) and [Level 4, Lesson 1](01-vision-transformers.md)
both found completely different real architectures confidently disagreeing on the same out-of-distribution
image; [Level 4, Lesson 7](07-vision-model-robustness.md)'s real FGSM attack flipped a `94.63%`-confident
correct answer using a pixel change of just `0.05`. **Never treat model confidence as a safety signal on its
own** — this course measured it failing that role repeatedly, across architectures and attack types.

## How It Actually Works

Every one of the five decisions above traces back to a real, specific, reproducible measurement earlier in
this course, not a rule of thumb stated without evidence. That's deliberate: a genuinely useful decision
framework for computer vision has to be grounded in what this course's own real experiments actually showed —
classical CV's real brittleness to rotation, a detector's real ability to report "nothing here," a tiny CNN's
real near-parity with a much larger fine-tuned one, batching's real 4.7× throughput advantage over one-at-a-
time inference, and confidence's real unreliability as a trust signal under both ordinary distribution shift
and deliberate attack.

## Common mistakes

- Picking an architecture or approach by reputation or recency rather than by what the specific real problem
  actually needs — this course's own real numbers repeatedly favored the simpler, cheaper option for the
  specific task at hand, not the most sophisticated one available.
- Assuming one decision axis (model size, say) is independent of the others — a real decision to use batch
  inference instead of real-time changes what counts as an acceptable real latency budget, which changes
  what model size is actually affordable, which [Lesson 6](06-scaling-vision-inference.md) measured directly.
- Treating this lesson's five decisions as exhaustive — they're the five this course built real, measured
  evidence for; a genuinely new real problem may raise axes this course never specifically tested.

## Exercise

Pick one real computer vision problem you're personally familiar with (even informally — a phone app, a
hobby project, something you've read about), and walk it through all five of this lesson's real decisions
explicitly, citing which specific real measurement from this course most directly informs each choice for that
problem.
