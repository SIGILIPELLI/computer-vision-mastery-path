---
title: "Computer Vision Interview Questions and Answers (179 Questions)"
description: "179 Computer Vision interview questions with answers and explanations, from beginner to advanced, organised by level. Free, from the Computer Vision Mastery Path."
---

# Computer Vision Interview Questions and Answers

**179 questions** with the correct answer and a short explanation, organised from beginner (Level 1) to advanced (Level 4). Every question comes from a lesson in the free Computer Vision Mastery Path, so you can open the lesson to go deeper.

!!! tip "Practise, don't just read"
    Each lesson also ends with an interactive quiz, and the [final quiz](quiz.md) tests all four levels. Try to answer each question below before reading the answer.

## Level 1 · Image Fundamentals & Classical CV

### [Images as Arrays: Pixels, Channels &amp; Color Spaces](level-1/01-images-as-arrays.md)

**Q1.** A color photo loaded as a NumPy array has shape (512, 512, 3). What does that mean?

**Answer:** 512 rows, 512 columns and 3 color channels per pixel

Images are 3-D arrays of integers (height, width, channels).

**Q2.** OpenCV loads color images as BGR while most other libraries assume RGB. What's the classic symptom if you forget?

**Answer:** Colors look wrong: reds and blues swapped

Convert with cv2.cvtColor when passing images between libraries.

**Q3.** What is the usual dtype and value range of an 8-bit image?

**Answer:** uint8, 0 to 255

Arithmetic on uint8 can overflow or wrap, so convert before math.

**Q4.** Why is HSV useful for color-based segmentation?

**Answer:** It separates hue from brightness, so a color can be selected more robustly under lighting changes

Hue/saturation/value are easier to threshold than RGB.

### [Image I/O, Resizing &amp; Color Space Conversions](level-1/02-image-io-resizing-colorspaces.md)

**Q5.** Which interpolation is the usual choice for shrinking an image?

**Answer:** INTER_AREA

Area averaging reduces aliasing when downscaling.

**Q6.** You resize 1000x500 to 200x200. What goes wrong?

**Answer:** The aspect ratio is distorted; compute the other dimension from the ratio

Preserve aspect ratio or pad.

**Q7.** Why measure interpolation quality with MSE against the original?

**Answer:** It gives a number for how much detail was lost after downscale-then-upscale

Lower error means closer to the original.

**Q8.** Does an RGB to HSV and back round trip return the original?

**Answer:** Essentially yes, within small rounding differences

Rounding to integers causes tiny differences.

### [Filtering &amp; Convolution: Blur, Sharpen, Edge Detection](level-1/03-filtering-convolution.md)

**Q9.** What does convolving an image with a 3x3 box kernel do?

**Answer:** Averages each pixel with its neighbours, blurring the image

The kernel slides across the image computing weighted sums.

**Q10.** Why did the hand-written convolution disagree with cv2.filter2D at first?

**Answer:** Border handling and correlation versus convolution details differ; padding mode must match

The lesson investigated and resolved it.

**Q11.** Which kernel type responds strongly at edges?

**Answer:** A gradient kernel such as Sobel

Edges are places where intensity changes quickly.

**Q12.** What does sharpening do conceptually?

**Answer:** Boosts high-frequency detail by subtracting a blurred version or using a center-weighted kernel

It amplifies local differences.

### [Thresholding &amp; Morphological Operations](level-1/04-thresholding-morphology.md)

**Q13.** What advantage does Otsu's method have over a fixed threshold of 127?

**Answer:** It computes a threshold from the image's own histogram instead of guessing

Data-computed beats guessed.

**Q14.** When is adaptive thresholding better than a global threshold?

**Answer:** When lighting varies across the image

It computes a threshold per neighbourhood.

**Q15.** What does morphological opening do?

**Answer:** Erosion followed by dilation, removing small isolated noise

Closing fills small holes.

**Q16.** What's morphological closing for?

**Answer:** Filling small gaps and holes inside foreground objects

Dilation followed by erosion.

### [Contours &amp; Shape Analysis](level-1/05-contours-shape-analysis.md)

**Q17.** Why did area filtering alone not give the right coin count?

**Answer:** Non-coin blobs of similar area passed, so a shape measure like circularity was needed

Combine area and circularity filters.

**Q18.** What does cv2.RETR_EXTERNAL return?

**Answer:** Only the outermost contours

Others retrieve hierarchy.

**Q19.** What does circularity (4*pi*area/perimeter^2) equal for a perfect circle?

**Answer:** 1

Lower values mean less round shapes.

**Q20.** What does the raw contour count include besides real objects?

**Answer:** Noise blobs and fragments, which is why filtering matters

The lesson started at 33 raw contours.

### [Feature Detection: Corners, Keypoints &amp; Descriptors](level-1/06-feature-detection.md)

**Q21.** What is a good keypoint?

**Answer:** A distinctive location (corner or blob) that can be found again in another view

Descriptors then describe it for matching.

**Q22.** What does ORB provide that Harris corners alone don't?

**Answer:** Keypoints plus binary descriptors for matching

Harris just finds corner responses.

**Q23.** The lesson measured ORB under 45-degree rotation and 2x scale. Why measure instead of assume?

**Answer:** Textbook invariance claims can be weaker in practice, especially for large scale changes

Measure robustness.

**Q24.** What distinguishes SIFT from ORB?

**Answer:** SIFT uses floating-point descriptors and is heavier; ORB is fast with binary descriptors

Trade accuracy against speed.

### [Feature Matching &amp; Homography: Aligning Images](level-1/07-feature-matching-homography.md)

**Q25.** What does RANSAC do in findHomography?

**Answer:** Fits the transform to inlier matches while ignoring outlier (wrong) matches

Random sampling finds a consensus model.

**Q26.** What does a homography represent?

**Answer:** A 3x3 projective transform mapping points between two views of a plane

Used for stitching and document alignment.

**Q27.** Why compare the recovered homography to a known ground truth?

**Answer:** To measure how accurately the algorithm recovers a transform it was never told

Known truth gives real accuracy.

**Q28.** What happens if many matches are wrong and you skip RANSAC?

**Answer:** The estimated transform can be badly distorted by outliers

Outliers need rejecting.

### [Image Augmentation for Training](level-1/08-image-augmentation.md)

**Q29.** Why use augmentation when training?

**Answer:** To expose the model to plausible variations and improve generalization

Flips, crops and jitter simulate real variation.

**Q30.** What does SSIM measure?

**Answer:** Structural similarity between two images (1 is identical)

It's a standard image-similarity metric.

**Q31.** What does cutout do?

**Answer:** Masks a random patch so the model can't rely on one region

It acts as a regularizer.

**Q32.** When is a horizontal flip a bad augmentation?

**Answer:** When orientation carries meaning, such as digits or text

Choose augmentations that preserve the label.

### [Classical Object Detection: Template Matching &amp; Haar Cascades](level-1/09-classical-object-detection.md)

**Q33.** What did the lesson find about Haar cascades in OpenCV 5.0?

**Answer:** This install no longer exposes CascadeClassifier

An honest, current finding.

**Q34.** Where does template matching break?

**Answer:** With scale and rotation changes of the target

It compares raw pixels at a fixed scale and orientation.

**Q35.** What does cv2.matchTemplate return?

**Answer:** A similarity map whose peak marks the best location

Use minMaxLoc to find the peak.

**Q36.** Why is template matching dangerous when pushed beyond its limits?

**Answer:** It can return a confident-looking location that is wrong

Always sanity-check peak scores.

### [Project: a Real Classical CV Pipeline (Document Scanner)](level-1/10-project-document-scanner.md)

**Q37.** What are the document scanner's main steps?

**Answer:** Edges, find the page contour, order its four corners, perspective-warp, then threshold

A classical pipeline.

**Q38.** Why order the four corners consistently?

**Answer:** The perspective transform needs matched source and destination corner order

Typically top-left, top-right, bottom-right, bottom-left.

**Q39.** The project used a synthetic scene. What does the lesson say about that?

**Answer:** It states plainly that the input is synthetic while the pipeline is real

Honesty about inputs.

**Q40.** What does getPerspectiveTransform produce?

**Answer:** A 3x3 matrix mapping four source points to four destination points

warpPerspective applies it.

## Level 2 · CNNs & Image Classification

### [Why CNNs: Convolution, Pooling &amp; Receptive Fields](level-2/01-why-cnns.md)

**Q41.** Why can a Conv2d layer reproduce a Sobel filter?

**Answer:** A convolution layer applies a learned kernel; setting its weights to Sobel gives the same operation

CNNs learn kernels instead of hand-designing them.

**Q42.** For input size W, kernel K, padding P and stride S, what's the output size?

**Answer:** floor((W - K + 2P) / S) + 1

The standard formula.

**Q43.** What does max pooling provide?

**Answer:** Down-sampling and some tolerance to small shifts

It keeps the strongest response in each window.

**Q44.** What is a receptive field?

**Answer:** The region of the input that influences a unit; it grows with stacked layers

Deeper layers see more.

### [Building a CNN From Scratch in PyTorch](level-2/02-building-a-cnn.md)

**Q45.** Why does weight sharing make convolutions parameter-efficient?

**Answer:** The same kernel is reused at every position instead of one weight per pixel connection

Compare a conv layer with an equivalent fully connected one.

**Q46.** How do you count a Conv2d layer's parameters?

**Answer:** out_channels * (in_channels * kh * kw) + out_channels bias terms

E.g. 1-&gt;8 channels, 3x3 kernel: 8*9+8 = 80.

**Q47.** Why run a random batch through a new model first?

**Answer:** To confirm tensor shapes before training

Shape errors show up immediately.

**Q48.** What does flattening do before fully connected layers?

**Answer:** Reshapes feature maps into a vector

The linear layer expects a fixed size.

### [Training a Real Image Classifier](level-2/03-training-a-real-classifier.md)

**Q49.** Why evaluate on a test set the model never saw?

**Answer:** Training accuracy can be inflated by memorization; test accuracy estimates generalization

Held-out data is the honest estimate.

**Q50.** Why did the lesson switch from CIFAR-10 to FashionMNIST?

**Answer:** The CIFAR-10 mirror was too slow to download in that environment

The lesson states this honestly.

**Q51.** What does the loss function do during training?

**Answer:** Measures how wrong predictions are so gradients can update weights

Cross-entropy is typical for classification.

**Q52.** Why shuffle training data each epoch?

**Answer:** So batches aren't ordered and the model doesn't learn order artefacts

Shuffling improves SGD behaviour.

### [Data Augmentation &amp; Regularization for Vision Models](level-2/04-augmentation-regularization.md)

**Q53.** What gap signals overfitting?

**Answer:** Training accuracy much higher than test accuracy

The gap grows over epochs without regularization.

**Q54.** What does dropout do?

**Answer:** Randomly zeroes activations during training to reduce co-adaptation

Disabled at evaluation.

**Q55.** What did the lesson learn about heavy regularization on this dataset?

**Answer:** It can trade off accuracy by underfitting; regularization isn't free

Measure the trade-off.

**Q56.** Why seed both models identically?

**Answer:** So differences come from regularization, not random initialization

A fair comparison.

### [CNN Architectures: ResNet &amp; Pretrained Backbones](level-2/05-cnn-architectures-resnet.md)

**Q57.** What problem do residual (skip) connections address?

**Answer:** Training very deep networks by letting gradients flow through identity shortcuts

They ease optimization.

**Q58.** ResNet-50 vs ResNet-18: what differs?

**Answer:** ResNet-50 is deeper with many more parameters

Counted directly in the lesson.

**Q59.** Both ResNets predicted something wrong for the astronaut photo. What does the lesson conclude?

**Answer:** ImageNet has no 'person' class, so they're forced to pick something else: out-of-distribution input

Reported honestly.

**Q60.** What is a pretrained backbone?

**Answer:** A network already trained on a large dataset (e.g. ImageNet), reused as a feature extractor

Basis for transfer learning.

### [Transfer Learning &amp; Fine-Tuning a Pretrained Model](level-2/06-transfer-learning.md)

**Q61.** What's frozen-backbone transfer learning?

**Answer:** Keep pretrained layers fixed and train only the new final layer

Fast but limited.

**Q62.** When does full fine-tuning tend to beat frozen features?

**Answer:** When the target domain differs and you have enough data to adapt all layers

Compare the two runs.

**Q63.** Why adapt ResNet-18 for FashionMNIST input?

**Answer:** The dataset is grayscale (1 channel) while ResNet expects 3

Change the first conv or replicate channels.

**Q64.** What is a sensible lr for fine-tuning pretrained layers?

**Answer:** Smaller than training from scratch, to avoid destroying useful features

Gentle updates preserve learned features.

### [Evaluating Vision Models: Confusion Matrices &amp; Per-Class Metrics](level-2/07-evaluating-vision-models.md)

**Q65.** Why isn't a single accuracy figure enough?

**Answer:** It can hide that some classes are much worse than others

Use per-class metrics.

**Q66.** What does a confusion matrix show?

**Answer:** How often each true class is predicted as each class

Off-diagonal entries are mistakes.

**Q67.** Precision for a class means...

**Answer:** Of everything predicted as that class, the share that was right

Recall is the share of true members found.

**Q68.** Why reload the saved checkpoint to re-evaluate?

**Answer:** To independently confirm the number instead of trusting the training script's print

Independent verification.

### [Visualizing What a CNN Learned](level-2/08-visualizing-cnn-features.md)

**Q69.** What does an input-gradient saliency map show?

**Answer:** Which input pixels most affect the prediction

Raw gradients can be noisy.

**Q70.** What is Grad-CAM-style visualization based on?

**Answer:** Last convolutional layer activations weighted by gradients, giving a coarse heat map

It highlights regions.

**Q71.** Why check attention maps against real pixel content?

**Answer:** A heat map story can sound plausible but not hold up (e.g. attention on background)

Quantify center versus border.

**Q72.** What was the lesson's first result?

**Answer:** Surprising, so it was investigated rather than accepted

Honest reporting.

### [Class Imbalance &amp; Hard Examples in Vision](level-2/09-class-imbalance-vision.md)

**Q73.** A class has 30 training images versus ~6,000 for the others. What's the expected effect?

**Answer:** The minority class is poorly learned and under-predicted with a standard loss

The model optimizes the majority.

**Q74.** What does a class-weighted loss do?

**Answer:** Penalizes minority-class mistakes more heavily

It can raise minority recall.

**Q75.** What's a risk of heavy class weights?

**Answer:** More false positives for the minority class (lower precision)

Look at precision and recall.

**Q76.** Other fixes for imbalance include...

**Answer:** Resampling and targeted augmentation of the minority class

Combine approaches.

### [Project: a Real Image Classifier Trained End to End](level-2/10-project-real-classifier.md)

**Q77.** How was the capstone configuration chosen?

**Answer:** From earlier findings: heavy augmentation and dropout underfit, so use a lighter touch

Evidence-based.

**Q78.** What does independent final evaluation add?

**Answer:** An unbiased estimate on held-out test data

Don't tune on the test set.

**Q79.** Why track train and test curves every epoch?

**Answer:** To see overfitting or underfitting as it develops

Diagnose, don't guess.

**Q80.** What's flip-only augmentation's rationale for FashionMNIST?

**Answer:** Horizontal flips preserve clothing labels

Label-preserving.

## Level 3 · Detection, Segmentation & Motion

### [Object Detection Fundamentals: Boxes, IoU &amp; Anchors](level-3/01-object-detection-fundamentals.md)

**Q81.** What is IoU?

**Answer:** Intersection over Union: overlap area divided by combined area of two boxes

1 is a perfect match.

**Q82.** Two 10x10 boxes overlap by half (x from 0-10 and 5-15). What's the IoU?

**Answer:** 1/3 (intersection 50, union 150)

Half overlap is only IoU 0.33.

**Q83.** What are anchors?

**Answer:** Predefined boxes of several scales and ratios that a detector refines

The standard set has 9 anchors.

**Q84.** Why verify IoU against hand-computed cases first?

**Answer:** Wrong IoU would quietly corrupt every later metric

Trust but verify.

### [Running &amp; Evaluating a Real Object Detector](level-3/02-running-a-real-detector.md)

**Q85.** Faster R-CNN trained on COCO detects how many classes?

**Answer:** 91 category ids (80 used) in COCO's label space

The lesson cites 91 classes.

**Q86.** What does a detection output include?

**Answer:** Boxes, class labels and confidence scores

Filter by score threshold.

**Q87.** How should a detector behave on a blank image?

**Answer:** Ideally return no confident detections, unlike a classifier that always picks a class

Detectors can say nothing.

**Q88.** Why threshold detections by score?

**Answer:** To drop low-confidence boxes that are likely false positives

Trade recall for precision.

### [Semantic Segmentation Fundamentals](level-3/03-semantic-segmentation-fundamentals.md)

**Q89.** What does semantic segmentation output?

**Answer:** A class label for every pixel

A dense prediction.

**Q90.** Why isn't pixel accuracy enough?

**Answer:** Large background classes dominate, hiding bad performance on small objects

Per-class IoU is stricter.

**Q91.** How is per-class IoU computed?

**Answer:** Pixels correct for the class divided by the union of predicted and true pixels of that class

Mean IoU averages classes.

**Q92.** Why use tiny hand-verifiable examples?

**Answer:** So every number can be checked by eye

Verify metrics before trusting them.

### [Running a Real Segmentation Model](level-3/04-running-a-real-segmentation-model.md)

**Q93.** DeepLabV3 outputs what compared with a detector?

**Answer:** A per-pixel class map instead of bounding boxes

argmax over the class channel.

**Q94.** How do you get the mask from output logits?

**Answer:** argmax over the class dimension

Each pixel gets the highest-scoring class.

**Q95.** How many classes does the VOC-trained model use?

**Answer:** 21 (20 objects plus background)

PASCAL VOC.

**Q96.** What does segmentation give that detection doesn't?

**Answer:** The object's exact shape, not just a rectangle

At more compute cost.

### [Instance Segmentation vs Semantic Segmentation](level-3/05-instance-vs-semantic-segmentation.md)

**Q97.** Two people stand side by side. What does semantic segmentation say?

**Answer:** Both are 'person' pixels with no separation between individuals

It labels classes, not instances.

**Q98.** What does Mask R-CNN add?

**Answer:** A separate mask per detected instance

It separates the two people.

**Q99.** When do you need instance segmentation?

**Answer:** When you must count or track individual objects of the same class

Semantic can't count touching objects.

**Q100.** What's the cost of instance segmentation?

**Answer:** More compute and a more complex model

Choose per need.

### [Pose Estimation &amp; Keypoint Detection](level-3/06-pose-estimation-keypoints.md)

**Q101.** How many keypoints does COCO pose use?

**Answer:** 17

Nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.

**Q102.** The astronaut photo is cropped at chest height. What did confidence scores reveal?

**Answer:** Which keypoints were actually visible versus guessed

Low scores mean forced guesses.

**Q103.** Why filter keypoints by confidence?

**Answer:** Low-confidence points can be wildly wrong

Don't draw guesses as facts.

**Q104.** What can pose estimation feed?

**Answer:** Activity recognition and gesture analysis

Keypoint sequences describe motion.

### [Optical Flow: Measuring Motion Between Frames](level-3/07-optical-flow.md)

**Q105.** What does optical flow estimate?

**Answer:** Apparent motion of pixels between frames

A motion vector per point or pixel.

**Q106.** What's the difference between Lucas-Kanade and dense flow?

**Answer:** Lucas-Kanade tracks sparse points; dense flow estimates motion for every pixel

Choose by need.

**Q107.** Why use a known exact pixel shift as a test?

**Answer:** It provides ground truth to measure algorithm accuracy

Real video rarely gives truth.

**Q108.** What assumption does flow rely on?

**Answer:** Brightness constancy and small motion between frames

It fails with large jumps.

### [Object Tracking Across Video Frames](level-3/08-object-tracking.md)

**Q109.** What did OpenCV 5.0 no longer expose?

**Answer:** The classic CSRT and KCF trackers

Same kind of removal as Haar cascades.

**Q110.** Why was an identical-looking distractor placed close to the target?

**Answer:** To test whether tracking survives visual ambiguity

Appearance trackers can jump.

**Q111.** What helps trackers survive ambiguity?

**Answer:** Motion models and re-detection or association logic

Combine detection and tracking.

**Q112.** Why use a synthetic video with a known trajectory?

**Answer:** You can measure tracking error exactly

Ground truth.

### [OCR: Detecting &amp; Recognizing Text in Images](level-3/09-ocr-text-detection-recognition.md)

**Q113.** What's the first OCR check in the lesson?

**Answer:** Exact match on clean rendered text, to establish a baseline

Then degrade systematically.

**Q114.** Which degradations did the lesson apply?

**Answer:** Blur, rotation and noise, measuring character accuracy

Find where OCR breaks.

**Q115.** Does OCR degrade uniformly?

**Answer:** No, different degradations break it at different points

That's why you measure.

**Q116.** How do you improve OCR before recognition?

**Answer:** Deskew, denoise and threshold the image

Preprocessing matters.

### [Project: a Real Multi-Stage Vision Pipeline](level-3/10-project-multi-stage-pipeline.md)

**Q117.** What are the pipeline's stages?

**Answer:** Detect the sign, track it across frames, then OCR its text

Detect, track, read.

**Q118.** Why track instead of detecting every frame?

**Answer:** Tracking is cheaper and keeps identity across frames

Detect periodically, track between.

**Q119.** The video is synthetic. Is that disclosed?

**Answer:** Yes: stages are real, the input video is synthetic, stated plainly

Honesty.

**Q120.** What can go wrong in multi-stage pipelines?

**Answer:** Errors compound: a bad detection ruins tracking and OCR

Evaluate each stage and the whole.

## Level 4 · Modern Architectures & Production CV

### [Vision Transformers: Concepts &amp; Real Inference](level-4/01-vision-transformers.md)

**Q121.** How does ViT-B/16 turn an image into tokens?

**Answer:** It splits it into 16x16 patches and embeds each patch linearly

Patches become a sequence.

**Q122.** The astronaut photo is out-of-distribution for ImageNet classifiers. Why?

**Answer:** ImageNet has no 'person' class, so any prediction is forced into another class

The coffee cup is in-distribution ('espresso').

**Q123.** What's a trade-off of ViTs versus CNNs?

**Answer:** ViTs have fewer inductive biases and usually need more data or pretraining

CNNs bake in locality.

**Q124.** What does self-attention let a ViT do?

**Answer:** Relate every patch to every other patch globally

Global context from the first layer.

### [Image Embeddings &amp; Similarity Search](level-4/02-image-embeddings-similarity-search.md)

**Q125.** What's an image embedding?

**Answer:** A vector from a network's penultimate layer capturing visual features

Used for similarity.

**Q126.** What does cosine similarity compare?

**Answer:** The angle between two vectors, ignoring length

1 means the same direction.

**Q127.** Why can nearest neighbours look counter-intuitive?

**Answer:** Embeddings capture what the network learned, which may not match human notions of similarity

The lesson investigated an unexpected pattern.

**Q128.** How would you scale similarity search to millions of images?

**Answer:** An approximate nearest-neighbour index (e.g. a vector database)

Exact scan is O(n) per query.

### [Self-Supervised &amp; Few-Shot Vision Concepts](level-4/03-self-supervised-few-shot.md)

**Q129.** How does a prototypical classifier classify with few labels?

**Answer:** Average each class's support embeddings into a prototype and assign the nearest

It reuses learned embeddings.

**Q130.** Why is it 'remarkably effective' here?

**Answer:** Good pretrained embeddings already separate classes; few examples just locate prototypes

The embedding does the work.

**Q131.** What's self-supervised learning?

**Answer:** Learning representations from unlabeled data using a pretext task

E.g. contrastive learning.

**Q132.** When does few-shot fail?

**Answer:** When embeddings don't capture the distinction the new classes need

Domain gaps hurt.

### [Image Generation Basics: GANs &amp; Diffusion](level-4/04-image-generation-basics.md)

**Q133.** In a GAN, what do the two networks do?

**Answer:** The generator makes fakes; the discriminator tries to tell fake from real

They're trained adversarially.

**Q134.** The lesson's discriminator was still winning after 10 epochs. What does that suggest?

**Answer:** Training hadn't reached balance, so generation quality was still limited

An honest trajectory.

**Q135.** Why evaluate the generator with an independent trained classifier?

**Answer:** Judging by eye is subjective; a classifier gives a quantitative signal

Independent evaluation.

**Q136.** What's the core idea of diffusion models?

**Answer:** Learn to reverse a gradual noising process to generate images

Iterative denoising.

### [Evaluating Generative &amp; Retrieval Vision Systems](level-4/05-evaluating-generative-retrieval.md)

**Q137.** Precision@k means...

**Answer:** Of the top k results, the share that are relevant

Recall@k is the share of all relevant items found in the top k.

**Q138.** Recall@k means...

**Answer:** Of all relevant items, the share found in the top k

Both depend on k.

**Q139.** What is mode collapse?

**Answer:** A generator producing a few repeated patterns instead of varied outputs

Check diversity.

**Q140.** How did the lesson measure diversity?

**Answer:** Pairwise embedding distances among generated images

Quantitative, not visual.

### [Scaling Vision Inference: Batching &amp; Latency](level-4/06-scaling-vision-inference.md)

**Q141.** Why does batching raise throughput?

**Answer:** Hardware processes many images in parallel, amortizing overhead

256 single calls versus one batched call.

**Q142.** What's the trade-off of large batches?

**Answer:** Higher per-request latency and diminishing returns past a peak

The lesson measured a peak.

**Q143.** Why measure rather than assume throughput?

**Answer:** Hardware and model details create real peaks and plateaus

Measure, don't assume.

**Q144.** Which metric matters for an interactive app?

**Answer:** Tail latency (e.g. p95) for single requests

Users feel latency.

### [Vision Model Robustness &amp; Common Failure Modes](level-4/07-vision-model-robustness.md)

**Q145.** What is an FGSM attack?

**Answer:** Adding a small perturbation in the direction of the loss gradient's sign to fool the model

Tiny, imperceptible changes can flip predictions.

**Q146.** What does adversarial vulnerability imply?

**Answer:** Models can fail on inputs humans find nearly identical

Robustness isn't accuracy.

**Q147.** Which common failure mode is distribution shift?

**Answer:** Test data differing from training (lighting, camera, domain)

Validate on realistic conditions.

**Q148.** How can you improve robustness?

**Answer:** Augmentation, adversarial training and testing under realistic perturbations

Measure robustness explicitly.

### [Bias &amp; Fairness Considerations in Computer Vision](level-4/08-bias-fairness-in-cv.md)

**Q149.** What's a well-documented bias mechanism the lesson tested?

**Answer:** Differential accuracy under varied conditions such as brightness

Performance gaps across groups or conditions.

**Q150.** Why report accuracy per subgroup?

**Answer:** An overall average can hide large gaps for some groups

Disaggregate.

**Q151.** What's a common source of bias in vision systems?

**Answer:** Unrepresentative training data

Data coverage matters.

**Q152.** What should you do when you find a gap?

**Answer:** Document it, improve data/model and decide whether the use is appropriate

Scale care with harm.

### [Choosing the Right CV Approach for a Real Problem](level-4/09-choosing-the-right-approach.md)

**Q153.** When is a classical CV pipeline the right choice?

**Answer:** Controlled setting, simple well-defined task and little data (e.g. a document scanner)

Simple beats complex when it works.

**Q154.** When do you reach for a pretrained deep model?

**Answer:** Complex variation and enough data or a good pretrained backbone

Transfer learning is the default.

**Q155.** What should you try first on a new problem?

**Answer:** The simplest baseline that could work, then measure

Establish a baseline.

**Q156.** What factors guide the choice?

**Answer:** Data available, accuracy needs, latency, hardware and maintainability

Match method to constraints.

### [Capstone: a Real End-to-End Computer Vision Application](level-4/10-capstone-end-to-end-cv-application.md)

**Q157.** What does the capstone combine?

**Answer:** A trained classifier within an automated quality-control pipeline

End to end.

**Q158.** Why define metrics before deployment?

**Answer:** So pass/fail decisions and costs of errors are explicit

False passes and false rejects cost differently.

**Q159.** How should a QC system handle low-confidence items?

**Answer:** Route them to human review

Use confidence thresholds.

**Q160.** What should you monitor after deployment?

**Answer:** Input drift and accuracy against periodic labeled samples

Models degrade as data shifts.

## Mixed questions across all levels

**Q161.** A photo loaded with OpenCV is shown with a blue face when plotted with another library. What's the cause?

**Answer:** OpenCV loads BGR while the plotting library expects RGB; convert with cvtColor

Channel order mismatch.

**Q162.** You shrink a 4000x3000 photo to thumbnails. Which interpolation is usual?

**Answer:** INTER_AREA

It reduces aliasing when downscaling.

**Q163.** Lighting varies across a scanned page. Which thresholding fits best?

**Answer:** Adaptive thresholding (a threshold per neighbourhood)

Global thresholds fail with uneven lighting.

**Q164.** Counting coins: area filtering still counts non-coin blobs. What extra check helps?

**Answer:** Circularity (4πA/P²) close to 1

Combine area and shape.

**Q165.** Why does RANSAC matter when estimating a homography from matched keypoints?

**Answer:** It rejects outlier matches so wrong pairs don't distort the transform

Fit to inliers.

**Q166.** Template matching works on a test image but fails on a zoomed copy. Why?

**Answer:** It compares raw pixels at fixed scale and rotation, so it isn't scale or rotation invariant

Use features or learned detectors.

**Q167.** A 3x3 conv with padding 1 and stride 1 on a 28x28 input gives what spatial size?

**Answer:** 28x28

floor((28-3+2)/1)+1 = 28.

**Q168.** Training accuracy is 99% and test accuracy 82%. What is happening and what helps?

**Answer:** Overfitting; add augmentation, dropout or more data

A large train-test gap signals overfitting.

**Q169.** You fine-tune a pretrained ResNet on grayscale FashionMNIST. What must change?

**Answer:** The first layer must accept 1 channel (or replicate channels) and the head must match 10 classes

Match the input and output shapes.

**Q170.** Overall accuracy is 94% but the 'sandal' class has 30 training images. What should you check?

**Answer:** Per-class precision and recall, since accuracy can hide a neglected minority class

Use a confusion matrix.

**Q171.** A heat map highlights the image background for a clothing label. What does it suggest?

**Answer:** The model may rely on spurious context; investigate the data

Attention maps must be checked against content.

**Q172.** Two people touch in a photo and you need to count them. Which model type?

**Answer:** Instance segmentation (e.g. Mask R-CNN)

Semantic masks merge same-class pixels.

**Q173.** A segmentation model has 95% pixel accuracy but ignores a small object class. Which metric exposes it?

**Answer:** Per-class IoU (mean IoU)

Large background classes dominate pixel accuracy.

**Q174.** Optical flow on frames where the object moves 80 pixels. What assumption is violated?

**Answer:** Small motion between frames; use pyramids or larger-range methods

LK assumes small displacements.

**Q175.** OCR accuracy drops sharply on a rotated receipt. What do you do?

**Answer:** Deskew (correct rotation) before recognition

Preprocess first.

**Q176.** You need to retrieve similar product photos from 5 million images. What's the standard approach?

**Answer:** Compute embeddings and use an approximate nearest-neighbour index

Exact scans are too slow.

**Q177.** A GAN generator's samples all look alike. What's this called and how do you measure it?

**Answer:** Mode collapse; measure diversity with pairwise embedding distances

Diversity must be quantified.

**Q178.** Latency matters for a live inspection line. Which metric should you watch?

**Answer:** Per-request tail latency (p95/p99), not only average throughput

Batching raises throughput but can raise latency.

**Q179.** A QC classifier is 99% accurate on lab images but degrades on the factory floor. What's the likely cause?

**Answer:** Distribution shift (lighting, camera, background); collect and validate on production-like data

Test under realistic conditions.

## Practical coding questions

### Intersection over Union

Implement iou(a, b) for boxes given as (x1, y1, x2, y2). Return intersection area divided by union area, and 0.0 when the boxes don't overlap.

Starter code:

```
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

Expected output:

```text
identical    1.000
half overlap 0.333
disjoint     0.000
contained    0.160
```

??? success "Show a solution"

    ```
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union else 0.0

    cases = [
        ("identical",    (0, 0, 10, 10), (0, 0, 10, 10)),
        ("half overlap", (0, 0, 10, 10), (5, 0, 15, 10)),
        ("disjoint",     (0, 0, 10, 10), (20, 20, 30, 30)),
        ("contained",    (0, 0, 10, 10), (2, 2, 6, 6)),
    ]
    for name, a, b in cases:
        print("%-12s %.3f" % (name, iou(a, b)))
    ```

    The overlap width and height are clamped at 0 for disjoint boxes. A half-overlapping pair has intersection 50 and union 150, so IoU is 1/3, not 1/2. Verified by running the solution.

### Per-class precision, recall and F1

From a confusion matrix (rows = true class, columns = predicted class) compute each class's precision, recall and F1, plus overall accuracy.

Starter code:

```
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

Expected output:

```text
accuracy 0.872
shirt   P=0.781 R=0.833 F1=0.806
sandal  P=0.938 R=0.857 F1=0.896
bag     P=0.909 R=0.900 F1=0.905
```

??? success "Show a solution"

    ```
    CLASSES = ["shirt", "sandal", "bag"]
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
    print("accuracy %.3f" % acc)
    for name, (p, r, f) in zip(CLASSES, per_class(CM)):
        print("%-7s P=%.3f R=%.3f F1=%.3f" % (name, p, r, f))
    ```

    Precision divides a class's diagonal entry by its column sum, recall by its row sum. Accuracy hides that 'shirt' has the weakest precision. Verified by running the solution.

---

More free practice: the [Computer Vision Mastery Path](index.md) curriculum, the [Interview Prep Pack](https://learnmasterypath.com/interview-pack/) and the [free resume ATS check](https://learnmasterypath.com/resume/).
