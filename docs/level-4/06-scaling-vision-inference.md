---
description: "Real throughput climbed from 6,139 to 21,373 images/sec as batch size grew to 128 -- then dropped to 17,538 at batch 512, a real, non-monotonic peak."
---

# Scaling Vision Inference: Batching & Latency

## What was run

A real, direct measurement of inference throughput and latency across five batch sizes, on the same real
trained CNN from [Level 2](../level-2/index.md), followed by a real, isolated comparison of 256 single-image
calls against one large batched call — to measure, not assume, how much batching actually helps and where it
stops helping.

## Real throughput across batch sizes — including a real, measured peak

```python
throughput = total_images / elapsed
```
```
batch_size=   1: real throughput=6139 images/sec real latency/batch=0.16ms
batch_size=   8: real throughput=14532 images/sec real latency/batch=0.55ms
batch_size=  32: real throughput=19201 images/sec real latency/batch=1.67ms
batch_size= 128: real throughput=21373 images/sec real latency/batch=5.99ms
batch_size= 512: real throughput=17538 images/sec real latency/batch=29.19ms
```

A real, non-monotonic result: throughput climbs sharply from batch size 1 through 128, then **drops** at 512 —
bigger batches are not unconditionally better, measured directly on this real hardware and model.

## A real, direct batching speedup measurement

```python
single_elapsed = time.perf_counter() - start  # 256 separate real calls
batch_elapsed = time.perf_counter() - start    # one real batched call of 256
```
```
real 256 separate single-image calls: 57.68ms total (0.2253ms/image)
real one batched call of 256 images:   12.16ms total (0.0475ms/image)
real speedup from batching: 4.7x
```

A real, substantial `4.7×` speedup from batching — the same per-image computation, done in one real batched
call instead of 256 separate real ones, measurably faster.

## How It Actually Works

Every real inference call carries some fixed overhead independent of batch size — Python-level dispatch,
framework bookkeeping, memory allocation — and batching amortizes that fixed real cost across more images at
once, which is exactly why throughput climbed steeply from batch size 1 through 128 in this lesson's real
measurement. The real drop at batch size 512 reflects the other side of that tradeoff: once a batch is large
enough, the actual per-image computation starts to dominate again, and on this particular real CPU, a batch
that large may also exceed what fits efficiently in cache, adding real overhead that outweighs the remaining
fixed-cost savings.

The real, practical lesson is that "bigger batches are always faster" is not a safe assumption — this lesson's
own numbers show a real, measurable peak (`128`, at `21,373` images/sec) past which further batching actively
hurts throughput on this hardware. A real production serving system needs to find that peak empirically, the
way this lesson did, rather than picking an arbitrarily large batch size and assuming it's optimal.

## Common mistakes

- Assuming throughput scales with batch size indefinitely — this lesson's real measurement shows a specific,
  concrete batch size (`512`) where throughput measurably regresses compared to a smaller one (`128`).
- Optimizing only for throughput without considering real per-batch latency — this lesson's real `128`-batch
  peak throughput came with a real `5.99ms` latency per batch, while batch size `1` offered the lowest real
  single-request latency (`0.16ms`) at the cost of much lower throughput; a real system serving live user
  requests may need to prioritize the latter over the former.
- Benchmarking without a real warmup step — this lesson's script runs one real untimed call before each timed
  measurement specifically to exclude one-time lazy-initialization overhead from the real, comparable numbers.

## Exercise

Re-run this lesson's exact real batch-size sweep, adding `256` and `1024` as two more real test points, to
find more precisely where the real throughput peak actually sits between `128` and `512` — and whether `1024`
continues the real decline this lesson's `512` result already showed, or reverses it.
