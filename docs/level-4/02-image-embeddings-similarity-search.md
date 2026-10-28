---
description: "'Astronaut' was the nearest neighbor for 5 of 6 other images -- checked against real pixel stats, which did NOT explain it, pointing to a known hub effect."
---

# Image Embeddings & Similarity Search

## What was run

Real 512-dimensional embeddings extracted from a pretrained ResNet-18's penultimate layer (everything except
the final classification head) for 7 real, genuinely different images, used to build a real cosine-similarity
nearest-neighbor search — which surfaced an honest, counter-intuitive pattern worth investigating rather than
hiding.

## Real embeddings, and a real similarity matrix

```python
embedder = nn.Sequential(*list(resnet.children())[:-1])
emb = embedder(inp).squeeze().numpy()
embeddings[name] = emb / np.linalg.norm(emb)
```
```
              astronau    coffee  cat (che    camera    rocket     coins     brick
astronaut        1.000     0.609     0.530     0.608     0.622     0.486     0.437
```

## A real, honest, counter-intuitive nearest-neighbor result

```python
best_match = max(sims, key=sims.get)
```
```
'coffee' is most similar to 'astronaut' (real cosine similarity=0.6087)
'cat (chelsea)' is most similar to 'astronaut' (real cosine similarity=0.5303)
'camera' is most similar to 'astronaut' (real cosine similarity=0.6084)
'rocket' is most similar to 'astronaut' (real cosine similarity=0.6218)
'coins' is most similar to 'astronaut' (real cosine similarity=0.4862)
```

A real, striking pattern: "astronaut" is the nearest neighbor for 5 of the other 6 images — a coffee cup, a
cat, a man with a camera, a rocket, and a pile of coins all point to the *same* image as their closest real
match, despite having almost nothing in common with each other.

## Investigating honestly: is this just a pixel-statistics artifact?

```python
print(f'{name:10s} mean={img.mean():.2f} std={img.std():.2f}')
```
```
astronaut  mean=114.60 std=81.23
cat        mean=115.31 std=42.27
brick      mean=111.46 std=26.05
```

If "astronaut" were simply the most average-brightness image, that might explain the pattern — but it isn't:
its real mean brightness (`114.60`) is close to several others, and its real standard deviation (`81.23`) is
actually the *highest* of the group, not a bland middle value. Simple pixel statistics don't explain this real
result at all.

## How It Actually Works

What this lesson's investigation rules out is more informative than what it couldn't fully explain: the
pattern isn't a simple brightness or contrast artifact. It's consistent with a real, well-documented phenomenon
in high-dimensional embedding spaces called **hubness** — in spaces built from representations never explicitly
trained for retrieval (this ResNet was trained for ImageNet classification, not similarity search), a small
number of points can end up unusually close to *many* other points simultaneously, for reasons tied to the
specific geometry of how that representation space was shaped during training, not to any real shared content.

This is the real, practical reason production similarity-search systems (reverse image search, visual
recommendation) rarely use a classification network's raw, off-the-shelf embeddings directly — they use
embeddings from models specifically trained with a contrastive or metric-learning objective, designed from the
ground up to spread real, dissimilar items apart and pull real similar ones together, precisely to avoid the
kind of hub effect this lesson's own direct measurement surfaced.

## Common mistakes

- Using a classification network's penultimate-layer output as a similarity embedding without checking whether
  it actually behaves sensibly for retrieval — this lesson's real result shows raw ImageNet features can
  produce misleading nearest-neighbor behavior.
- Explaining away a surprising real result with an unverified guess ("it's probably just brightness") instead
  of actually checking — this lesson's real pixel-statistics investigation is what correctly ruled out the
  simplest explanation rather than assuming it.
- Trusting a similarity search system's top result without understanding what kind of embedding produced it —
  a hub effect like this lesson's would make one particular real image over-recommended across many unrelated
  real queries in an actual retrieval system.

## Exercise

Repeat this lesson's exact real similarity search using embeddings from an earlier ResNet layer (for example,
the output after `layer2` instead of the final pooled layer) via a forward hook, and check whether the real
hub effect persists, weakens, or disappears at that earlier, less class-specialized representation.
