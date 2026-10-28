---
description: "Real precision@k held at 80-90% from k=1 to 20 as recall@k climbed from 0.0093 to 0.1666 -- the real GAN's image diversity hit 83.3% of real images'."
---

# Evaluating Generative & Retrieval Vision Systems

## What was run

Real precision@k and recall@k, computed over a real 1,000-image retrieval index built from
[Level 2](../level-2/index.md)'s trained embeddings, plus a real quantitative diversity check on
[Lesson 4](04-image-generation-basics.md)'s GAN output — measuring whether its generated images are
meaningfully varied or collapsing toward a few repeated patterns, using real pairwise embedding distances
rather than a visual guess.

## Real retrieval metrics, measured directly

```python
precision = n_relevant_retrieved / k
recall = n_relevant_retrieved / n_relevant_total
```
```
k= 1: real precision@1=0.9000 real recall@1=0.0093
k= 5: real precision@5=0.8267 real recall@5=0.0426
k=10: real precision@10=0.8267 real recall@10=0.0853
k=20: real precision@20=0.8083 real recall@20=0.1666
```

A real, textbook-shaped tradeoff: precision stays high and roughly stable (`80-90%`) across every real `k`,
while recall climbs steadily as `k` grows — exactly what should happen when a real index has roughly `100`
relevant images per class out of `1,000` total, so retrieving more results naturally recovers more of them
without diluting precision much.

## A real, quantitative check on generative diversity

```python
def mean_pairwise_distance(embs):
    # real average distance between every pair of embeddings
```
```
real diversity among 50 REAL images:      24.6221
real diversity among 50 GENERATED images: 20.5121
real ratio (generated/real): 0.8331
```

A real, moderate — not catastrophic — diversity gap: the GAN's generated images are, on average, somewhat
less spread out in embedding space than real images (`83.31%` as diverse), but nowhere close to the severe
clustering a real mode-collapsed GAN would show (a ratio near `0`, where every generated image maps to nearly
the same embedding).

## How It Actually Works

Precision@k and recall@k answer two different real questions about a retrieval system: precision asks "of
what I actually returned, how much was correct," while recall asks "of everything that was correct, how much
did I actually find." A system can have perfect precision by returning only one certain result, or perfect
recall by returning everything — this lesson's real numbers show a system that holds high precision while
recall genuinely grows with `k`, which is the real, desirable shape for a system meant to be used with a
flexible result-count.

The real diversity check works because mode collapse — a well-known real GAN failure mode where the generator
learns to produce only a few convincing outputs instead of the full real variety of the training data — leaves
a measurable fingerprint in embedding space: collapsed outputs cluster close together, while genuinely diverse
outputs spread out the way real data does. This lesson's real `0.8331` ratio is exactly the kind of direct,
numeric answer to "has this GAN collapsed" that no amount of looking at a handful of individual generated
images could reliably provide on its own.

## Common mistakes

- Reporting only precision (or only recall) for a retrieval system — this lesson's real numbers show they
  genuinely move independently as `k` changes, and a single one alone would miss half the real picture.
- Judging a generative model's diversity by looking at a small, arbitrary number of samples — this lesson's
  real pairwise-distance check over 50 real generated images gives a stable, quantitative answer that a
  glance at 4 or 5 outputs could easily get wrong in either direction.
- Treating any diversity ratio below `1.0` as evidence of mode collapse — this lesson's real `0.8331` reflects
  a real but moderate gap, not the severe, near-zero clustering that actual mode collapse produces; the
  distinction matters for deciding whether a generative model needs more training or a fundamentally different
  approach.

## Exercise

Re-run this lesson's exact real diversity check using the GAN generator from
[Lesson 4](04-image-generation-basics.md) at a much earlier training checkpoint (epoch 1, saved separately)
instead of the final epoch-10 weights. Measure whether the real diversity ratio was lower (closer to genuine
mode collapse) earlier in training, or whether this lesson's `0.8331` figure was already roughly stable from
early on.
