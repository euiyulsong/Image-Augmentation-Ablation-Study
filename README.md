# Image Augmentation Ablation Study

This repository contains a controlled ablation study on the effect of image augmentation during image-classification training.

The experiment compares three training regimes:

- **Original Only**: train only on the original images
- **Augmented Only**: train using augmented samples
- **Mixed**: train using both original and augmented samples

The main goal is to measure how these strategies affect:

1. Clean-image accuracy
2. Robustness to rotation
3. Robustness to occlusion
4. Robustness to low contrast
5. Robustness to dark images

---

## Experimental Setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 |
| Model | Vision Transformer (ViT) |
| Epochs | 10 |
| Batch size | 128 |
| Seed | 42 |
| Training regimes | Original Only / Augmented Only / Mixed |

Experiment command:

```bash
python3 augmentation_ablation.py \
  --model vit \
  --epochs 10 \
  --batch-size 128 \
  --seeds 42
```

---

## Training Regimes

### Original Only

The model is trained only on the original training images.

This serves as the baseline and measures how well a standard training setup generalizes to corrupted or transformed test images.

### Augmented Only

Training samples are passed through the augmentation pipeline.

The goal is to increase invariance to image transformations and improve robustness to distribution shifts.

### Mixed

The model is trained using a mixture of original and augmented images.

This setting attempts to preserve the original data distribution while also exposing the model to transformed samples.

---

# Results

## Overall Results

| Training regime | Clean | Rotation | Occlusion | Low Contrast | Dark | Robust Avg ↑ | Robust Drop ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original Only | 97.66% | 91.87% | 89.14% | 97.20% | 97.22% | 93.86% | 3.80 pp |
| Augmented Only | 98.32% | 93.29% | 89.81% | 97.95% | 98.08% | 94.78% | 3.54 pp |
| **Mixed** | **98.44%** | **93.83%** | **90.74%** | **98.05%** | **98.09%** | **95.18%** | **3.26 pp** |

`Robust Avg` is the average performance across:

- Rotation
- Occlusion
- Low contrast
- Dark

`Robust Drop` is:

```text
Clean Accuracy - Robust Average
```

Lower is better.

---

## Improvement over Original-Only Training

| Metric | Original Only | Mixed | Improvement |
|---|---:|---:|---:|
| Clean | 97.66% | **98.44%** | **+0.78 pp** |
| Rotation | 91.87% | **93.83%** | **+1.96 pp** |
| Occlusion | 89.14% | **90.74%** | **+1.60 pp** |
| Low Contrast | 97.20% | **98.05%** | **+0.85 pp** |
| Dark | 97.22% | **98.09%** | **+0.87 pp** |
| Robust Average | 93.86% | **95.18%** | **+1.32 pp** |
| Robust Drop | 3.80 pp | **3.26 pp** | **-0.54 pp** |

The largest gains are observed under **rotation** and **occlusion**, which are also the most difficult perturbations in this experiment.

---

# Key Findings

## 1. Mixed training achieves the best overall performance

The mixed training regime achieves the highest score on **every evaluated condition**:

```text
Clean          : 98.44%
Rotation       : 93.83%
Occlusion      : 90.74%
Low Contrast   : 98.05%
Dark           : 98.09%
Robust Average : 95.18%
```

Compared with original-only training, the robust average improves from:

```text
93.86% → 95.18%
```

an absolute improvement of approximately:

```text
+1.32 percentage points
```

---

## 2. Augmentation does not hurt clean accuracy in this experiment

A common concern is that aggressive augmentation may improve robustness at the cost of in-distribution accuracy.

That trade-off was not observed here.

```text
Original Only : 97.66%
Augmented Only: 98.32%
Mixed         : 98.44%
```

Both augmentation regimes outperform the original-only baseline on clean CIFAR-10 test images.

The mixed regime gives the highest clean accuracy.

---

## 3. Rotation benefits the most from augmentation

Rotation accuracy improves substantially:

```text
Original Only : 91.87%
Augmented Only: 93.29%
Mixed         : 93.83%
```

Mixed training improves rotation robustness by:

```text
+1.96 percentage points
```

relative to the original-only model.

This is the largest absolute improvement among the tested perturbations.

---

## 4. Occlusion remains the hardest perturbation

All models perform substantially worse under occlusion:

```text
Original Only : 89.14%
Augmented Only: 89.81%
Mixed         : 90.74%
```

Although mixed training improves the result by **+1.60 pp**, occlusion remains the lowest-performing condition.

This suggests that partial information loss is harder for the classifier to handle than moderate appearance changes such as brightness or contrast shifts.

---

## 5. Low contrast and darkness have relatively small effects

The models remain highly accurate under low-contrast and dark-image conditions.

For mixed training:

```text
Clean        : 98.44%
Low Contrast : 98.05%
Dark         : 98.09%
```

The degradation relative to clean accuracy is small compared with rotation or occlusion.

This indicates that the trained ViT is relatively robust to the tested photometric perturbations.

---

## 6. Mixed training produces the smallest robustness gap

Robust drop:

```text
Original Only : 3.80 pp
Augmented Only: 3.54 pp
Mixed         : 3.26 pp
```

Therefore, mixed training reduces the difference between clean and corrupted-image performance.

Compared with the original-only baseline:

```text
3.80 pp → 3.26 pp
```

which corresponds to a reduction of:

```text
0.54 percentage points
```

in the clean-to-robustness gap.

---

# Training Curves

## Original Only

| Epoch | Train Loss | Train Accuracy | Validation Accuracy |
|---:|---:|---:|---:|
| 1 | 0.3455 | 91.84% | 97.08% |
| 2 | 0.0573 | 98.45% | 97.56% |
| 3 | 0.0223 | 99.50% | 97.28% |
| 4 | 0.0090 | 99.87% | 97.26% |
| 5 | 0.0042 | 99.97% | 97.80% |
| 6 | 0.0025 | 99.98% | 97.86% |
| 7 | 0.0017 | 99.99% | **97.88%** |
| 8 | 0.0013 | 99.99% | 97.86% |
| 9 | 0.0011 | 100.00% | 97.84% |
| 10 | 0.0011 | 100.00% | 97.82% |

The original-only model reaches almost perfect training accuracy, while validation accuracy saturates around 97.8%.

---

## Augmented Only

| Epoch | Train Loss | Train Accuracy | Validation Accuracy |
|---:|---:|---:|---:|
| 1 | 0.4057 | 89.69% | 96.80% |
| 2 | 0.1072 | 96.69% | 97.50% |
| 3 | 0.0752 | 97.66% | 97.68% |
| 4 | 0.0554 | 98.24% | 97.94% |
| 5 | 0.0442 | 98.53% | 97.96% |
| 6 | 0.0353 | 98.96% | 98.16% |
| 7 | 0.0285 | 99.10% | **98.34%** |
| 8 | 0.0236 | 99.32% | **98.34%** |
| 9 | 0.0217 | 99.38% | 98.28% |
| 10 | 0.0199 | 99.48% | **98.34%** |

Augmentation makes the training task harder, resulting in higher training loss and lower training accuracy than original-only training, while validation performance improves.

---

## Mixed

| Epoch | Train Loss | Train Accuracy | Validation Accuracy |
|---:|---:|---:|---:|
| 1 | 0.3803 | 90.63% | 97.18% |
| 2 | 0.0890 | 97.31% | 97.48% |
| 3 | 0.0580 | 98.27% | 97.76% |
| 4 | 0.0428 | 98.71% | 98.16% |
| 5 | 0.0310 | 99.12% | 98.02% |
| 6 | 0.0235 | 99.32% | 98.10% |
| 7 | 0.0213 | 99.40% | 97.98% |
| 8 | 0.0179 | 99.50% | **98.40%** |
| 9 | 0.0160 | 99.56% | 98.28% |
| 10 | 0.0149 | 99.58% | 98.28% |

The mixed regime reaches the highest observed validation accuracy of **98.40%**.

---

# Summary

The experiment gives the following ordering:

```text
Mixed > Augmented Only > Original Only
```

for both clean accuracy and average robustness.

The most important result is:

```text
                    Clean       Robust Avg
Original Only       97.66%        93.86%
Augmented Only      98.32%        94.78%
Mixed               98.44%        95.18%
```

Under this experimental configuration, combining original and augmented examples gives the best balance between clean accuracy and robustness.

The improvement is especially visible for geometric or information-removing perturbations:

```text
Rotation:
91.87% → 93.83% (+1.96 pp)

Occlusion:
89.14% → 90.74% (+1.60 pp)
```

These results support using a **mixture of original and augmented samples** rather than relying exclusively on either distribution.

---

# Important Limitation

The current experiment uses only:

```text
seed = 42
```

Therefore, these numbers represent a **single experimental run**.

Standard deviation cannot be estimated from one seed, which is why the current STD values are `NaN`.

For a more reliable comparison, run multiple seeds, for example:

```bash
python3 augmentation_ablation.py \
  --model vit \
  --epochs 10 \
  --batch-size 128 \
  --seeds 42 43 44
```

Then report:

```text
mean ± std
```

for each metric.

Until multiple-seed experiments are performed, differences such as `+0.12 pp` or `+0.78 pp` should be interpreted as observed differences rather than evidence of statistically significant improvements.
