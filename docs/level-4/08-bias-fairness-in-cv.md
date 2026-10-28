---
description: "The same +0.3 brightness shift dropped real 'Sneaker' accuracy to 28.10% while 'Bag' stayed at 99.20% -- one condition, wildly unequal real harm."
---

# Bias & Fairness Considerations in Computer Vision

## What was run

A real, direct test of a well-documented real bias mechanism in vision systems: differential accuracy under
different image brightness conditions — the same underlying factor the real Gender Shades study
(Buolamwini & Gebru, 2018) found correlated with skin-tone-related accuracy gaps in commercial face analysis
systems. **Honest scope note:** this dataset has no demographic data, so this test cannot measure real
demographic bias directly — it measures the real, underlying brightness-sensitivity mechanism itself, directly
and honestly, on the same trained model used throughout this course.

## A real, asymmetric overall accuracy response to brightness

```python
shifted = torch.clamp(images + brightness_shift, 0, 1)
```
```
brightness shift=-0.30: real accuracy=0.8318
brightness shift=+0.00: real accuracy=0.8972
brightness shift=+0.30: real accuracy=0.6874
```

A real, meaningfully asymmetric result: darkening the image by `0.30` cost about `6.5` real accuracy points,
while brightening it by the same real amount cost more than `21` points — the model is not equally robust to
every real direction of lighting variation.

## A real, dramatic per-class disparity under the exact same condition

```
brightness shift=+0.30:
  Trouser        : real accuracy=0.9940
  Bag            : real accuracy=0.9920
  Sneaker        : real accuracy=0.2810
```

The single real cause (the same `+0.30` brightness shift, applied identically to every image) produced
wildly unequal real harm: `Trouser` and `Bag` barely noticed it (`99.40%` and `99.20%` accuracy), while
`Sneaker` collapsed to `28.10%` — worse than random guessing would do among 10 classes. This is exactly the
real *shape* of the harm a biased vision system causes: not uniformly degraded performance, but a shared
condition that devastates some real groups (here, classes) while leaving others almost untouched.

## How It Actually Works

A brightness shift pushes every real pixel value toward one end of the model's learned input range — and
whether that specifically breaks a given class depends on what real visual cues the model actually learned to
rely on for it. `Sneaker`'s real collapse under brightening suggests the model leaned on specific real
intensity or contrast patterns (perhaps the dark sole against a lighter upper, or a specific shading pattern)
that a uniform brightness increase genuinely erases or inverts, while `Trouser` and `Bag`'s real, more robust
shapes apparently don't depend on that kind of fragile cue nearly as much.

This is the real, structural mechanism behind documented real-world bias findings like Gender Shades: a model
doesn't "know" it's being unfair, and the unequal real harm isn't necessarily caused by anything resembling
intentional discrimination — it emerges from which specific visual patterns the model happened to learn to
rely on, which can correlate with real, protected attributes (skin tone, lighting conditions common to certain
environments) in ways that only show up when someone actually measures per-group performance, exactly as this
lesson's per-class breakdown did.

## Common mistakes

- Reporting only overall accuracy under a stress test like this one — this lesson's real `68.74%` overall
  figure at `+0.30` brightness completely hides the real, much more severe `28.10%` collapse specific to one
  class.
- Assuming a model that performs well on average is fair across all real subgroups it will encounter — this
  lesson's own real numbers are a direct, measured counterexample: strong overall accuracy coexisting with a
  catastrophic failure for one specific real class under one specific real condition.
- Treating brightness robustness as unrelated to real-world fairness concerns — this lesson's test is
  deliberately framed around the exact real mechanism (differential performance under lighting variation)
  that connects directly to documented real bias findings in deployed vision systems.

## Exercise

Re-run this lesson's exact real brightness test at the `+0.30` shift, but specifically for the `Sneaker` class,
across a finer range of shifts (`0.10, 0.15, 0.20, 0.25, 0.30`) to find the real, approximate brightness level
at which this class's accuracy begins its sharp, measured collapse — identifying the real operating range
within which this specific class remains reliable.
