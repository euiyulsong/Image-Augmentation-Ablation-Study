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

CNN에서는 augmentation 효과가 매우 뚜렷하다. 특히 **occlusion에서 +18.52%p**, rotation에서 **+6.51%p** 개선되어 augmentation이 단순 clean accuracy보다 robustness 향상에 큰 효과를 보였다. 

---

## 3. ViT 결과

| Training | Clean | Rotation | Occlusion | Low Contrast | Dark | Robust Avg | Robust Drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original only | 97.58% | 88.64% | 90.93% | 96.91% | 96.87% | 93.34% | 4.24%p |
| **Augmented only** | **98.02%** | **90.93%** | **91.64%** | 96.96% | **97.33%** | **94.22%** | **3.80%p** |
| Mixed | 97.97% | 90.58% | 91.18% | **97.16%** | 97.29% | 94.05% | 3.92%p |



### Augmentation 효과

`original_only → augmented_only`

| Metric | 변화 |
|---|---:|
| Clean | **+0.44%p** |
| Rotation | **+2.29%p** |
| Occlusion | **+0.71%p** |
| Low contrast | **+0.05%p** |
| Dark | **+0.46%p** |
| Robust Avg | **+0.88%p** |
| Robust Drop | **4.24 → 3.80%p** |

ViT에서도 augmentation이 전반적으로 개선을 만들었지만, CNN과 비교하면 효과 크기는 훨씬 작았다. 이미 `original_only` 상태에서 높은 robustness를 보였기 때문에 추가 augmentation의 marginal gain이 상대적으로 작다. 

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
| ViT | **98.02%** | **90.93%** | **91.64%** | **94.22%** |
| VLM | **98.02%** | **90.93%** | **91.64%** | **94.22%** |

가장 큰 차이는 **augmentation을 하지 않았을 때의 robustness**에서 나타난다.

- CNN original-only Robust Avg: **84.78%**
- ViT/VLM original-only Robust Avg: **93.34%**

즉 이번 실험에서는 ViT/VLM 계열이 augmentation 없이도 상대적으로 강한 robustness를 보였으며, CNN은 augmentation으로 그 격차를 상당 부분 줄였다. 

---

# 6. 주요 결론

### 1. Image augmentation은 모든 모델에서 성능을 떨어뜨리지 않았다

이번 실험에서는 `augmented_only`가 세 모델 모두 clean accuracy를 유지하거나 개선했다.

```text
CNN : 95.65 → 96.36 (+0.71%p)
ViT : 97.58 → 98.02 (+0.44%p)
VLM : 97.58 → 98.02 (+0.44%p)
```

따라서 augmentation으로 인한 clean performance degradation은 관찰되지 않았다.

### 2. CNN에서 augmentation 효과가 특히 크다

CNN Robust Avg:

```text
Original   84.78%
   ↓
Mixed      91.51%
   ↓
Augmented  92.11%
```

특히 occlusion:

```text
68.65% → 87.17%
         +18.52%p
```

따라서 CNN에서는 augmentation이 robustness 확보에 매우 중요하게 작용했다.

### 3. ViT/VLM에서는 augmentation 효과가 상대적으로 작다

ViT/VLM Robust Avg:

```text
93.34 → 94.22
        +0.88%p
```

이미 baseline robustness가 높기 때문에 augmentation의 추가 효과가 CNN보다 작게 나타났다.

### 4. Mixed보다 Augmented-only가 대부분 약간 우수했다

이번 실험에서는 예상과 달리 `original + augmented`를 섞은 mixed보다 `augmented_only`가 대부분 조금 더 높았다.

```text
CNN Robust Avg
Augmented : 92.11
Mixed     : 91.51

ViT/VLM Robust Avg
Augmented : 94.22
Mixed     : 94.05
```

다만 차이가 0.2~0.6%p 수준이고 seed가 하나뿐이므로 **현재 결과만으로 augmented-only가 mixed보다 우수하다고 결론내리기는 어렵다.**

---

# 7. 중요한 실험상 주의점

## ViT와 VLM 결과가 완전히 동일함

현재 로그에서 ViT와 VLM은 **epoch별 loss/train accuracy/validation accuracy뿐 아니라 모든 최종 metric까지 정확하게 동일하다.**

예를 들어 VLM augmented-only:

```text
clean       0.9802
rotate      0.9093
occlusion   0.9164
robust_avg  0.9422
```

ViT augmented-only 역시 정확히 같은 값이다. 

이는 독립적인 두 모델의 결과라면 매우 이례적이므로 다음을 확인할 필요가 있다.

```text
--model vlm
--model vit
```

분기가 실제로 서로 다른 backbone/model을 생성하는지 확인해야 한다. 동일한 encoder 또는 동일한 코드 path를 사용하고 있을 가능성이 있다.

또한 현재 **seed=42 하나뿐**이라 STD가 모두 `NaN`이다. 따라서 augmentation 효과가 작은 ViT/VLM에서는 최소 3 seeds 정도를 돌린 뒤 평균 ± 표준편차로 비교하는 것이 적절하다.

---

# 최종 요약

> **이번 CIFAR-10 실험에서는 image augmentation이 clean accuracy를 유지하면서 corruption robustness를 개선했다. 효과는 CNN에서 가장 컸으며, 특히 occlusion 성능이 68.65%에서 87.17%로 +18.52%p 증가했다. ViT/VLM에서도 augmentation이 robustness를 개선했지만 개선 폭은 약 +0.88%p로 상대적으로 작았다. 다만 ViT와 VLM 결과가 완전히 동일하므로 실제로 서로 다른 모델이 실행되었는지 코드 검증이 필요하며, seed가 하나뿐이므로 추가 seed 실험 후 통계적 비교가 필요하다.**
