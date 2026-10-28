---
description: "With only 30 real minority examples, plain training got 40.1% recall on that class; a class-weighted loss raised it to 78.3%, and overall accuracy too."
---

# Class Imbalance & Hard Examples in Vision

## What was run

A real, deliberately imbalanced FashionMNIST training set — only 30 real "Sandal" images against the full
~6,000 real examples of every other class — trained two real ways: with a standard, unweighted loss, and with
a real class-weighted loss that penalizes minority-class mistakes more heavily, to measure exactly what happens
to the minority class in each case.

## A real, severely imbalanced training set

```python
minority_idx = np.where(targets == MINORITY_CLASS)[0][:MINORITY_COUNT]
```
```
REAL imbalanced training set: 30 real 'Sandal' examples vs 54000 examples of everything else
```

A real ratio of 1 minority example for every 1,800 majority examples — a genuinely severe, not subtle,
imbalance.

## Real plain training: high overall accuracy, hiding a real minority-class failure

```python
criterion = nn.CrossEntropyLoss()  # no weighting
```
```
real overall accuracy: 0.8372
REAL 'Sandal' recall: 0.4010
```

A real, respectable-looking `83.72%` overall accuracy — while the model correctly identifies real Sandals only
`40.10%` of the time, missing the majority of them. The overall number, dominated by the nine well-represented
classes, almost completely hides this real, specific failure.

## A real fix, and a real, uncomplicated win

```python
weights[MINORITY_CLASS] = len(majority_idx) / MINORITY_COUNT / 9
criterion = nn.CrossEntropyLoss(weight=weights)
```
```
real class weight assigned to minority class: 200.0

real overall accuracy: 0.8662
REAL 'Sandal' recall: 0.7830
```

A real, substantial improvement on both real numbers at once: minority-class recall nearly doubled
(`0.4010 → 0.7830`), and — unlike [Lesson 4](04-augmentation-regularization.md)'s regularization tradeoff —
overall accuracy *also* improved (`0.8372 → 0.8662`). Fixing the minority class's real weakness didn't cost
anything measurable elsewhere this time.

## How It Actually Works

A standard loss function treats every real training example equally, which means a class with only 30 real
examples contributes a vanishingly small fraction of the total real gradient signal across an entire epoch —
the model has very little real incentive to get those 30 examples right when getting the other 53,970 right
already produces a low average loss. Weighting the minority class's loss contribution by a real factor of
`200` (roughly matching how outnumbered it actually is) directly counteracts that imbalance, forcing a real
mistake on a Sandal to cost the model as much as 200 real mistakes on anything else would.

The real reason this fix produced a clean win, rather than [Lesson 4](04-augmentation-regularization.md)'s
tradeoff, is that this problem wasn't really about the model's total capacity — it was about *where* that
capacity's incentive was pointed. Reweighting the loss doesn't reduce the model's ability to fit real patterns
anywhere; it just redirects attention toward a real class the model was previously almost able to ignore.

## Common mistakes

- Reporting overall accuracy on an imbalanced real dataset without checking per-class recall — this lesson's
  real `83.72%` overall figure looked perfectly reasonable while hiding a genuine `40.10%` recall failure on
  the one class that actually mattered for this investigation.
- Assuming class weighting always trades accuracy elsewhere for minority-class improvement — this lesson's
  real result improved both simultaneously, because the fix addressed a genuine training-signal imbalance, not
  a capacity limit.
- Picking an arbitrary class weight without grounding it in the real imbalance ratio — this lesson's real
  weight of `200` was computed directly from the actual real 1,800:1 minority-to-majority ratio (divided across
  9 majority classes), not guessed.

## Exercise

Repeat this lesson's exact real experiment with an even smaller real minority count (`MINORITY_COUNT = 10`
instead of `30`) and check whether the real class-weighted fix still improves both overall accuracy and
minority recall simultaneously, or whether a real imbalance this severe starts requiring a tradeoff the way
[Lesson 4](04-augmentation-regularization.md)'s regularization did.
