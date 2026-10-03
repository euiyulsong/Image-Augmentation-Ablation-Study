# Image Augmentation Ablation 결과

## 1. 실험 설정

- Dataset: **CIFAR-10**
- Epochs: **10**
- Batch size: **128**
- Seed: **42**
- 비교 모델:
  - CNN
  - ViT
  - VLM
- 학습 regime:
  - `original_only`: 원본 이미지만 사용
  - `augmented_only`: augmentation 적용 이미지 사용
  - `mixed`: 원본 + augmentation 혼합
- Robustness 평가:
  - Rotation
  - Occlusion
  - Low contrast
  - Dark

현재 실험은 seed 42 하나만 실행되어 STD는 계산되지 않았다.

---

## 2. CNN 결과

| Training | Clean | Rotation | Occlusion | Low Contrast | Dark | Robust Avg | Robust Drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original only | 95.65% | 83.85% | 68.65% | 92.58% | 94.04% | 84.78% | 10.87%p |
| **Augmented only** | **96.36%** | **90.36%** | **87.17%** | **95.21%** | **95.69%** | **92.11%** | **4.25%p** |
| Mixed | 96.28% | 89.71% | 85.98% | 94.79% | 95.55% | 91.51% | 4.77%p |

### Augmentation 효과

`original_only → augmented_only`

| Metric | 변화 |
|---|---:|
| Clean | **+0.71%p** |
| Rotation | **+6.51%p** |
| Occlusion | **+18.52%p** |
| Low contrast | **+2.63%p** |
| Dark | **+1.65%p** |
| Robust Avg | **+7.33%p** |
| Robust Drop | **10.87 → 4.25%p** |

CNN에서는 augmentation 효과가 매우 뚜렷하다. 특히 **occlusion에서 +18.52%p**, rotation에서 **+6.51%p** 개선되어 augmentation이 단순 clean accuracy뿐 아니라 robustness 향상에도 큰 효과를 보였다.

---

## 3. ViT 결과

| Training | Clean | Rotation | Occlusion | Low Contrast | Dark | Robust Avg | Robust Drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original only | 97.66% | 91.87% | 89.14% | 97.20% | 97.22% | 93.86% | 3.80%p |
| Augmented only | 98.32% | 93.29% | 89.81% | 97.95% | 98.08% | 94.78% | 3.54%p |
| **Mixed** | **98.44%** | **93.83%** | **90.74%** | **98.05%** | **98.09%** | **95.18%** | **3.26%p** |

### Augmentation 효과

`original_only → augmented_only`

| Metric | 변화 |
|---|---:|
| Clean | **+0.66%p** |
| Rotation | **+1.42%p** |
| Occlusion | **+0.67%p** |
| Low contrast | **+0.75%p** |
| Dark | **+0.86%p** |
| Robust Avg | **+0.92%p** |
| Robust Drop | **3.80 → 3.54%p** |

ViT에서도 augmentation이 모든 평가 항목에서 성능을 개선했다.

특히 이번 ViT 실험에서는 `augmented_only`보다 **`mixed`가 모든 metric에서 가장 높은 성능**을 기록했다.

`original_only → mixed` 기준으로 보면:

| Metric | 변화 |
|---|---:|
| Clean | **+0.78%p** |
| Rotation | **+1.96%p** |
| Occlusion | **+1.60%p** |
| Low contrast | **+0.85%p** |
| Dark | **+0.87%p** |
| Robust Avg | **+1.32%p** |
| Robust Drop | **3.80 → 3.26%p** |

즉 ViT에서는 원본 데이터와 augmentation 데이터를 함께 사용하는 전략이 clean accuracy와 robustness 모두에서 가장 좋은 결과를 보였다.

---

## 4. VLM 결과

| Training | Clean | Rotation | Occlusion | Low Contrast | Dark | Robust Avg | Robust Drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original only | 97.58% | 88.64% | 90.93% | 96.91% | 96.87% | 93.34% | 4.24%p |
| **Augmented only** | **98.02%** | **90.93%** | **91.64%** | 96.96% | **97.33%** | **94.22%** | **3.80%p** |
| Mixed | 97.97% | 90.58% | 91.18% | **97.16%** | 97.29% | 94.05% | 3.92%p |

### Augmentation 효과

`original_only → augmented_only`

- Clean: **+0.44%p**
- Rotation: **+2.29%p**
- Occlusion: **+0.71%p**
- Robust Avg: **+0.88%p**
- Robust Drop: **4.24 → 3.80%p**

VLM에서도 augmentation이 소폭의 clean 성능 상승과 robustness 개선을 만들었다.

---

# 5. 모델 간 비교

Augmented-only 기준:

| Model | Clean | Rotation | Occlusion | Robust Avg |
|---|---:|---:|---:|---:|
| CNN | 96.36% | 90.36% | 87.17% | 92.11% |
| **ViT** | **98.32%** | **93.29%** | 89.81% | **94.78%** |
| VLM | 98.02% | 90.93% | **91.64%** | 94.22% |

Augmented-only 기준으로 ViT는 **Clean, Rotation, Robust Avg**에서 가장 높은 성능을 기록했고, VLM은 **Occlusion**에서 가장 높은 성능을 보였다.

Augmentation을 사용하지 않은 `original_only`의 Robust Avg는 다음과 같다.

```text
CNN : 84.78%
ViT : 93.86%
VLM : 93.34%
```

ViT와 VLM은 augmentation 없이도 CNN보다 높은 corruption robustness를 보였다.

CNN은 augmentation 적용 후 Robust Avg가 **84.78% → 92.11%**로 크게 증가하면서 격차를 상당 부분 줄였다.

ViT의 경우 augmented-only보다 mixed가 더 높은 결과를 기록했다.

```text
ViT Robust Avg

Original   : 93.86%
Augmented  : 94.78%
Mixed      : 95.18%
```

---

# 6. 주요 결론

### 1. Image augmentation은 모든 모델에서 clean accuracy를 유지하거나 개선했다

이번 실험에서는 augmentation을 적용한 학습이 세 모델 모두에서 clean accuracy를 떨어뜨리지 않았다.

```text
CNN
95.65 → 96.36 (+0.71%p)

ViT
97.66 → 98.32 (+0.66%p)

VLM
97.58 → 98.02 (+0.44%p)
```

따라서 현재 실험에서는 augmentation으로 인한 clean performance degradation은 관찰되지 않았다.

---

### 2. CNN에서 augmentation 효과가 가장 크다

CNN Robust Avg:

```text
Original   84.78%
Mixed      91.51%
Augmented  92.11%
```

특히 occlusion:

```text
68.65% → 87.17%
         +18.52%p
```

Rotation 역시:

```text
83.85% → 90.36%
         +6.51%p
```

CNN에서는 augmentation이 corruption robustness 확보에 매우 크게 작용했다.

---

### 3. ViT에서도 augmentation이 모든 robustness metric을 개선했다

ViT Robust Avg:

```text
Original   : 93.86%
Augmented  : 94.78%
Mixed      : 95.18%
```

`original_only → augmented_only`에서는 Robust Avg가 **+0.92%p** 증가했고, `original_only → mixed`에서는 **+1.32%p** 증가했다.

Rotation에서도:

```text
Original   : 91.87%
Augmented  : 93.29%
Mixed      : 93.83%
```

Occlusion에서도:

```text
Original   : 89.14%
Augmented  : 89.81%
Mixed      : 90.74%
```

augmentation에 따른 일관된 개선이 관찰되었다.

다만 CNN의 Robust Avg 개선 폭인 **+7.33%p**와 비교하면 ViT의 augmentation 효과는 상대적으로 작다. 이는 ViT의 baseline robustness 자체가 이미 높은 것과 관련이 있을 수 있다.

---

### 4. ViT에서는 Mixed가 가장 좋은 결과를 보였다

기존 결과와 달리 새 ViT 실험에서는 `mixed`가 `augmented_only`보다 모든 metric에서 높았다.

```text
                  Augmented   Mixed
Clean               98.32     98.44
Rotation            93.29     93.83
Occlusion           89.81     90.74
Low Contrast        97.95     98.05
Dark                98.08     98.09
Robust Avg          94.78     95.18
Robust Drop          3.54      3.26
```

따라서 ViT에서는 **원본과 augmented sample을 함께 사용하는 mixed training이 가장 좋은 결과**를 보였다.

반면 CNN과 VLM에서는 augmented-only가 mixed보다 소폭 높았다.

```text
CNN Robust Avg
Augmented : 92.11
Mixed     : 91.51

VLM Robust Avg
Augmented : 94.22
Mixed     : 94.05

ViT Robust Avg
Augmented : 94.78
Mixed     : 95.18
```

즉 augmentation regime의 최적 선택은 모델 architecture에 따라 달라질 가능성이 있다.

다만 현재는 seed가 하나뿐이기 때문에 작은 차이에 대한 일반적인 우열을 확정하기는 어렵다.

---

# 7. 중요한 실험상 주의점

## 현재 모든 실험은 seed 하나만 사용함

현재 실험은 모두 **seed=42 하나만 실행**되었기 때문에 STD가 `NaN`이다.

특히 ViT에서:

```text
Augmented Robust Avg : 94.78%
Mixed Robust Avg     : 95.18%
Difference           : +0.40%p
```

처럼 작은 차이는 random seed에 따라 순서가 달라질 수 있다.

따라서 보다 신뢰할 수 있는 비교를 위해서는 최소 3개 이상의 seed를 사용하는 것이 적절하다.

예:

```bash
--seeds 42 43 44
```

이후 결과를 다음과 같이 평균 ± 표준편차 형태로 비교하는 것이 좋다.

```text
Robust Avg = Mean ± STD
```

---

## ViT와 VLM 결과는 현재 서로 다름

새로운 ViT 실험 결과에서는 기존과 달리 ViT와 VLM 결과가 명확하게 다르게 나타난다.

예를 들어 `augmented_only`의 경우:

```text
             ViT      VLM
Clean        98.32    98.02
Rotation     93.29    90.93
Occlusion    89.81    91.64
Robust Avg   94.78    94.22
```

따라서 이전에 관찰되었던 **ViT/VLM 결과가 완전히 동일한 문제는 새 ViT 실행에서는 나타나지 않는다.**

다만 VLM 결과가 별도 VLM backbone을 실제로 사용해 얻은 결과인지는 코드의 `--model vlm` 분기를 확인해두는 것이 좋다.

---

# 최종 요약

> **이번 CIFAR-10 실험에서는 image augmentation이 CNN, ViT, VLM 모두에서 clean accuracy를 유지하거나 개선하면서 corruption robustness를 향상시켰다.**
>
> **CNN에서 augmentation 효과가 가장 컸으며**, Robust Avg가 **84.78%에서 92.11%로 +7.33%p** 증가했다. 특히 occlusion 성능은 **68.65%에서 87.17%로 +18.52%p** 개선되었다.
>
> **ViT에서는 augmentation 적용 시 Robust Avg가 93.86%에서 94.78%로 증가했고, mixed training에서는 95.18%까지 증가했다.** 또한 mixed가 clean, rotation, occlusion, low contrast, dark를 포함한 모든 metric에서 가장 높은 결과를 기록했다.
>
> **VLM에서도 augmentation을 통해 Robust Avg가 93.34%에서 94.22%로 개선되었다.**
>
> 전체적으로 ViT/VLM은 augmentation 없이도 CNN보다 높은 baseline robustness를 보였으며, CNN은 augmentation을 통해 그 격차를 크게 줄였다. 또한 augmentation regime의 최적 선택은 모델별로 달랐으며, CNN/VLM에서는 augmented-only가, ViT에서는 mixed가 가장 높은 Robust Avg를 기록했다.
>
> 다만 현재 모든 실험이 **seed=42 하나만 사용**했기 때문에 0.x%p 수준의 작은 차이는 통계적으로 확정하기 어렵다. 최소 3개 이상의 seed를 사용해 평균과 표준편차를 함께 비교하는 추가 실험이 필요하다.
