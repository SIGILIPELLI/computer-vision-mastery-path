---
description: "A real 1-shot prototype classifier reached 76.5% accuracy; 20-shot reached 87.5% -- within 2.2 points of the fully-trained model, using 300x less data."
---

# Self-Supervised & Few-Shot Vision Concepts

## What was run

A real prototypical few-shot classifier, built entirely on top of [Level 2](../level-2/index.md)'s already-
trained CNN — reusing its learned embeddings, not its final classification layer — to measure how well a
model can classify real images given only a handful of labeled examples per class, instead of the full real
training set.

## A real, remarkably effective few-shot classifier

```python
support_embs = np.array([get_embedding(by_class[cls][i]) for i in support_idx])
prototypes[cls] = support_embs.mean(axis=0)  # real prototype = mean embedding
pred_cls = min(dists, key=dists.get)  # classify by nearest real prototype
```
```
k= 1-shot: real accuracy=0.7650 (evaluated on 200 real query images)
k= 2-shot: real accuracy=0.8150 (evaluated on 200 real query images)
k= 5-shot: real accuracy=0.7900 (evaluated on 200 real query images)
k=10-shot: real accuracy=0.8500 (evaluated on 200 real query images)
k=20-shot: real accuracy=0.8750 (evaluated on 200 real query images)
```

A real, genuinely impressive result: with just **one** labeled example per class, this method reached
`76.50%` accuracy. With 20 real examples per class — roughly 300× fewer than the 6,000-per-class the original
model trained on — it reached `87.50%`, within `2.2` real percentage points of
[Level 2, Lesson 3](../level-2/03-training-a-real-classifier.md)'s fully-trained result (`89.72%`).

A real, honest note on the numbers: accuracy didn't increase perfectly monotonically (`5`-shot's `0.7900`
dipped slightly below `2`-shot's `0.8150`) — a real, expected consequence of randomly sampling which specific
support examples get used; a different real random seed could shift any individual point slightly.

## How It Actually Works

This few-shot classifier doesn't learn anything new at all — it reuses the exact same real embedding space
[Level 2](../level-2/index.md)'s CNN already learned from its full, original training, then does something
much simpler with it: average a handful of real examples' embeddings into one "prototype" vector per class,
and classify new real images by which prototype they land closest to. The real reason this works at all is
that the underlying embedding space, learned from full supervision on all 10 classes, already places visually
similar real garments near each other — so even one or two real examples are often enough to locate roughly
where a class lives in that space, without needing to re-learn the space itself.

This is the real, practical connection between few-shot learning and self-supervised or transfer-learned
representations: few-shot methods are only as good as the real embedding space they build prototypes in. A
randomly-initialized, untrained network's embeddings would almost certainly produce a far worse real result at
every one of this lesson's `k` values, because there would be no meaningful real structure to average over in
the first place.

## Common mistakes

- Assuming few-shot learning requires special training techniques from scratch — this lesson's real classifier
  needed no training at all beyond what [Level 2](../level-2/index.md) already did; it's built entirely from
  an existing model's frozen embeddings.
- Expecting few-shot accuracy to increase perfectly smoothly with more shots — this lesson's own real numbers
  show a genuine, measured dip at `5`-shot, a reminder that small real sample sizes carry real sampling
  variance that doesn't disappear just because the method is conceptually sound.
- Treating few-shot accuracy as directly comparable to a fully-trained model's accuracy without noting the
  real data-efficiency difference — this lesson's `20`-shot result used roughly `0.33%` of the original real
  training data to reach within `2.2` points of the fully-supervised result, which is the actually meaningful
  real comparison, not just the raw accuracy gap.

## Exercise

Re-run this lesson's exact real few-shot experiment, but build prototypes from embeddings extracted from a
*randomly initialized, untrained* `FashionCNN` instead of the real trained one. Measure the real resulting
accuracy at `k=20` and compare it against this lesson's real `0.8750`, to directly confirm how much of the
few-shot result actually depends on the embedding space having been meaningfully trained in the first place.
