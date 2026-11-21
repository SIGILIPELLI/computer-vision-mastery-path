---
title: "Final Quiz"
description: "Test what you have learned across all four levels of Computer Vision Mastery Path with a 19-question final quiz."
---

# Final Quiz

19 questions drawn from all four levels of this course. You need **70% (14 out of 19)** to pass —
you can retake it as many times as you like. Your best result is saved in this browser.

<div id="mp-quiz-root"></div>

<script type="application/json" id="mp-quiz-data">
[
  {
    "question": "A photo loaded with OpenCV is shown with a blue face when plotted with another library. What's the cause?",
    "options": [
      "The image is corrupt",
      "The resolution is low",
      "The model is wrong",
      "OpenCV loads BGR while the plotting library expects RGB; convert with cvtColor"
    ],
    "correctIndex": 3,
    "explanation": "Channel order mismatch."
  },
  {
    "question": "You shrink a 4000x3000 photo to thumbnails. Which interpolation is usual?",
    "options": [
      "None",
      "Random",
      "INTER_NEAREST",
      "INTER_AREA"
    ],
    "correctIndex": 3,
    "explanation": "It reduces aliasing when downscaling."
  },
  {
    "question": "Lighting varies across a scanned page. Which thresholding fits best?",
    "options": [
      "Adaptive thresholding (a threshold per neighbourhood)",
      "Fixed 127",
      "Otsu on the whole page",
      "No threshold"
    ],
    "correctIndex": 0,
    "explanation": "Global thresholds fail with uneven lighting."
  },
  {
    "question": "Counting coins: area filtering still counts non-coin blobs. What extra check helps?",
    "options": [
      "Color",
      "File name",
      "Circularity (4πA/P²) close to 1",
      "Image size"
    ],
    "correctIndex": 2,
    "explanation": "Combine area and shape."
  },
  {
    "question": "Why does RANSAC matter when estimating a homography from matched keypoints?",
    "options": [
      "It rejects outlier matches so wrong pairs don't distort the transform",
      "It resizes",
      "It speeds the camera",
      "It finds keypoints"
    ],
    "correctIndex": 0,
    "explanation": "Fit to inliers."
  },
  {
    "question": "Template matching works on a test image but fails on a zoomed copy. Why?",
    "options": [
      "It needs color",
      "It compares raw pixels at fixed scale and rotation, so it isn't scale or rotation invariant",
      "It needs a GPU",
      "It's buggy"
    ],
    "correctIndex": 1,
    "explanation": "Use features or learned detectors."
  },
  {
    "question": "A 3x3 conv with padding 1 and stride 1 on a 28x28 input gives what spatial size?",
    "options": [
      "30x30",
      "28x28",
      "14x14",
      "26x26"
    ],
    "correctIndex": 1,
    "explanation": "floor((28-3+2)/1)+1 = 28."
  },
  {
    "question": "Training accuracy is 99% and test accuracy 82%. What is happening and what helps?",
    "options": [
      "Underfitting; add layers",
      "Overfitting; add augmentation, dropout or more data",
      "Nothing",
      "A bug in the loader"
    ],
    "correctIndex": 1,
    "explanation": "A large train-test gap signals overfitting."
  },
  {
    "question": "You fine-tune a pretrained ResNet on grayscale FashionMNIST. What must change?",
    "options": [
      "Only lr",
      "Nothing",
      "The first layer must accept 1 channel (or replicate channels) and the head must match 10 classes",
      "Only the dataset"
    ],
    "correctIndex": 2,
    "explanation": "Match the input and output shapes."
  },
  {
    "question": "Overall accuracy is 94% but the 'sandal' class has 30 training images. What should you check?",
    "options": [
      "Per-class precision and recall, since accuracy can hide a neglected minority class",
      "Only loss",
      "Only speed",
      "Nothing"
    ],
    "correctIndex": 0,
    "explanation": "Use a confusion matrix."
  },
  {
    "question": "A heat map highlights the image background for a clothing label. What does it suggest?",
    "options": [
      "Nothing",
      "It's perfect",
      "The map is color",
      "The model may rely on spurious context; investigate the data"
    ],
    "correctIndex": 3,
    "explanation": "Attention maps must be checked against content."
  },
  {
    "question": "Two people touch in a photo and you need to count them. Which model type?",
    "options": [
      "Instance segmentation (e.g. Mask R-CNN)",
      "Edge detection",
      "Image classification",
      "Semantic segmentation"
    ],
    "correctIndex": 0,
    "explanation": "Semantic masks merge same-class pixels."
  },
  {
    "question": "A segmentation model has 95% pixel accuracy but ignores a small object class. Which metric exposes it?",
    "options": [
      "Throughput",
      "Per-class IoU (mean IoU)",
      "Loss",
      "Pixel accuracy"
    ],
    "correctIndex": 1,
    "explanation": "Large background classes dominate pixel accuracy."
  },
  {
    "question": "Optical flow on frames where the object moves 80 pixels. What assumption is violated?",
    "options": [
      "Aspect ratio",
      "Resolution",
      "Color constancy",
      "Small motion between frames; use pyramids or larger-range methods"
    ],
    "correctIndex": 3,
    "explanation": "LK assumes small displacements."
  },
  {
    "question": "OCR accuracy drops sharply on a rotated receipt. What do you do?",
    "options": [
      "Add noise",
      "Deskew (correct rotation) before recognition",
      "Shrink",
      "Convert to color"
    ],
    "correctIndex": 1,
    "explanation": "Preprocess first."
  },
  {
    "question": "You need to retrieve similar product photos from 5 million images. What's the standard approach?",
    "options": [
      "Compare pixels pairwise",
      "Retrain per query",
      "Sort by file name",
      "Compute embeddings and use an approximate nearest-neighbour index"
    ],
    "correctIndex": 3,
    "explanation": "Exact scans are too slow."
  },
  {
    "question": "A GAN generator's samples all look alike. What's this called and how do you measure it?",
    "options": [
      "Drift; measure fps",
      "Underfitting; measure lr",
      "Overfitting; measure loss",
      "Mode collapse; measure diversity with pairwise embedding distances"
    ],
    "correctIndex": 3,
    "explanation": "Diversity must be quantified."
  },
  {
    "question": "Latency matters for a live inspection line. Which metric should you watch?",
    "options": [
      "Per-request tail latency (p95/p99), not only average throughput",
      "Parameter count",
      "Disk size",
      "Throughput only"
    ],
    "correctIndex": 0,
    "explanation": "Batching raises throughput but can raise latency."
  },
  {
    "question": "A QC classifier is 99% accurate on lab images but degrades on the factory floor. What's the likely cause?",
    "options": [
      "Too few epochs",
      "Wrong loss",
      "GPU issue",
      "Distribution shift (lighting, camera, background); collect and validate on production-like data"
    ],
    "correctIndex": 3,
    "explanation": "Test under realistic conditions."
  }
]
</script>

## Coding challenges

Try these on your own computer, then compare your output. Nothing you write is sent anywhere.

### Intersection over Union

Implement iou(a, b) for boxes given as (x1, y1, x2, y2). Return intersection area divided by union area, and 0.0 when the boxes don't overlap.

**Starter code**

```python
def iou(a, b):
    # boxes are (x1, y1, x2, y2); return intersection area / union area
    return 0.0

cases = [
    ("identical",    (0, 0, 10, 10), (0, 0, 10, 10)),
    ("half overlap", (0, 0, 10, 10), (5, 0, 15, 10)),
    ("disjoint",     (0, 0, 10, 10), (20, 20, 30, 30)),
    ("contained",    (0, 0, 10, 10), (2, 2, 6, 6)),
]
for name, a, b in cases:
    print("%-12s %.3f" % (name, iou(a, b)))
```

**Expected output**

```text
identical    1.000
half overlap 0.333
disjoint     0.000
contained    0.160
```

<details><summary>Show solution and explanation</summary>

<pre><code>def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0

cases = [
    (&quot;identical&quot;,    (0, 0, 10, 10), (0, 0, 10, 10)),
    (&quot;half overlap&quot;, (0, 0, 10, 10), (5, 0, 15, 10)),
    (&quot;disjoint&quot;,     (0, 0, 10, 10), (20, 20, 30, 30)),
    (&quot;contained&quot;,    (0, 0, 10, 10), (2, 2, 6, 6)),
]
for name, a, b in cases:
    print(&quot;%-12s %.3f&quot; % (name, iou(a, b)))
</code></pre>

<p>The overlap width and height are clamped at 0 for disjoint boxes. A half-overlapping pair has intersection 50 and union 150, so IoU is 1/3, not 1/2. Verified by running the solution.</p>
</details>

### Per-class precision, recall and F1

From a confusion matrix (rows = true class, columns = predicted class) compute each class's precision, recall and F1, plus overall accuracy.

**Starter code**

```python
CLASSES = ["shirt", "sandal", "bag"]
# rows = true class, columns = predicted class
CM = [
    [50,  2,  8],
    [ 4, 30,  1],
    [10,  0, 90],
]

def per_class(cm):
    # return [(precision, recall, f1), ...] for each class, rows = true, cols = predicted
    return [(0.0, 0.0, 0.0)] * len(cm)

total = sum(sum(r) for r in CM)
acc = sum(CM[i][i] for i in range(len(CM))) / total
print("accuracy %.3f" % acc)
for name, (p, r, f) in zip(CLASSES, per_class(CM)):
    print("%-7s P=%.3f R=%.3f F1=%.3f" % (name, p, r, f))
```

**Expected output**

```text
accuracy 0.872
shirt   P=0.781 R=0.833 F1=0.806
sandal  P=0.938 R=0.857 F1=0.896
bag     P=0.909 R=0.900 F1=0.905
```

<details><summary>Show solution and explanation</summary>

<pre><code>CLASSES = [&quot;shirt&quot;, &quot;sandal&quot;, &quot;bag&quot;]
# rows = true class, columns = predicted class
CM = [
    [50,  2,  8],
    [ 4, 30,  1],
    [10,  0, 90],
]

def per_class(cm):
    n = len(cm)
    out = []
    for i in range(n):
        tp = cm[i][i]
        pred = sum(cm[r][i] for r in range(n))
        true = sum(cm[i])
        precision = tp / pred if pred else 0.0
        recall = tp / true if true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append((precision, recall, f1))
    return out

total = sum(sum(r) for r in CM)
acc = sum(CM[i][i] for i in range(len(CM))) / total
print(&quot;accuracy %.3f&quot; % acc)
for name, (p, r, f) in zip(CLASSES, per_class(CM)):
    print(&quot;%-7s P=%.3f R=%.3f F1=%.3f&quot; % (name, p, r, f))
</code></pre>

<p>Precision divides a class&#x27;s diagonal entry by its column sum, recall by its row sum. Accuracy hides that &#x27;shirt&#x27; has the weakest precision. Verified by running the solution.</p>
</details>
