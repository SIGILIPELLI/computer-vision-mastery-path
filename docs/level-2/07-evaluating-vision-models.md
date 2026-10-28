---
description: "Behind one real 89.72% accuracy number, per-class F1 ranged from 0.9849 (Trouser) to 0.7056 (Shirt) -- confused with T-shirt/top 121/1,000 times."
---

# Evaluating Vision Models: Confusion Matrices & Per-Class Metrics

## What was run

The exact real model trained in [Lesson 3](03-training-a-real-classifier.md), loaded back from its saved
checkpoint and re-evaluated completely independently — not trusting the training script's own printed number —
with a real confusion matrix and real per-class precision/recall/F1 computed to see what a single accuracy
figure actually hides.

## A real, independently-reconfirmed accuracy, then broken down by class

```python
model.load_state_dict(torch.load("fashion_cnn.pt", map_location=device))
overall_acc = (all_preds == all_labels).mean()
```
```
REAL overall test accuracy (recomputed independently from the saved model): 0.8972
```

The real `0.8972` matches [Lesson 3](03-training-a-real-classifier.md)'s original number exactly — confirming
the saved checkpoint and the reloaded model genuinely reproduce the same real behavior, not an assumption.

## The real confusion matrix: one accuracy number, ten very different stories

```python
cm = confusion_matrix(all_labels, all_preds)
```
```
              precision    recall  f1-score   support
 T-shirt/top     0.8455    0.8540    0.8498      1000
     Trouser     0.9939    0.9760    0.9849      1000
    Pullover     0.8875    0.7890    0.8354      1000
       Shirt     0.7380    0.6760    0.7056      1000
      Sandal     0.9732    0.9820    0.9776      1000
```

A real, dramatic spread hidden behind the single `0.8972` accuracy figure: `Trouser` is nearly perfect
(F1=`0.9849`), while `Shirt` is the model's clear weak point (F1=`0.7056`) — a real 28-point gap in F1 between
the model's best and worst real categories.

## A real, specific, named failure mode

```python
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
max_confusion_idx = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
```
```
REAL most-confused pair: true='Shirt' predicted='T-shirt/top' (121 real test images)
```

Of the real 1,000 true "Shirt" test images, 121 were specifically mislabeled as "T-shirt/top" — the single
largest real confusion anywhere in the entire matrix, and the direct explanation for why `Shirt`'s real recall
(`0.6760`) is so much lower than its peers.

## How It Actually Works

A single accuracy number averages correctness across every real class equally weighted by how often each
appears in the test set — it has no way to show that the model is nearly flawless on `Trouser` and `Sandal`
while genuinely struggling on `Shirt`. The confusion matrix instead shows every real (true class, predicted
class) pair's count directly, which is what let this lesson find not just *that* the model struggles with
shirts, but *specifically* that it struggles by confusing them with T-shirts — a real, visually sensible
mistake, since a collared shirt and a T-shirt in a small, grayscale, 28×28 image genuinely do share a lot of
real visual structure.

Precision and recall capture two different real failure directions a single accuracy score collapses into one
number: `Shirt`'s real recall of `0.6760` means the model misses a lot of real shirts (classifying them as
something else); its real precision of `0.7380` means that even when it *does* predict "Shirt," it's wrong
about 26% of the time. Both numbers matter, and a system that only reports overall accuracy would never reveal
that this specific, real, asymmetric weakness exists.

## Common mistakes

- Reporting a single accuracy number as a complete description of model quality — this lesson's real `0.8972`
  hid a genuine 28-point F1 gap between the best and worst-performing classes.
- Investigating low accuracy without looking at the confusion matrix's actual off-diagonal structure — this
  lesson's real investigation found the *specific* confusion (Shirt → T-shirt/top) driving the weak class's
  score, not just that the score was weak.
- Trusting a training script's self-reported final accuracy without an independent, separate re-evaluation —
  this lesson's real recomputation from the saved checkpoint, run as a completely separate script, is what
  actually confirms the number is trustworthy rather than a bug in how it was computed the first time.

## Exercise

Using this lesson's real confusion matrix, identify the second-most-confused real pair of classes (the next
largest off-diagonal count after Shirt→T-shirt/top), and look up both classes' real precision and recall to
predict, before checking, which of the two classes should show the larger real recall deficit.
