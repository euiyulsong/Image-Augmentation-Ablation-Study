| 계열 | 공식 예시 | 실제 augmentation | 강도 |
|---|---|---|---|
| Supervised ViT | DeiT | RandAugment + Mixup + CutMix + Random Erasing + ColorJitter | **강함** |
| Generative VLM | BLIP | RandomResizedCrop + Flip + RandAugment | 중간 |
| Generative VLM | BLIP-2 | RandomResizedCrop + Flip | 약함 |
| Generative VLM | Qwen3-VL FT | dynamic resize / pixel bounds 중심, explicit random aug 없음 | 매우 약함/없음 |
| Embedding VLM | original CLIP | resized image에서 random square crop | **약함** |
| Embedding VLM | OpenCLIP/CLIPA recipe | crop + ColorJitter + grayscale | 중간 |
| Embedding VLM | SigLIP official example | resize + flip + RandAugment | **상당히 강함** |
| Qwen3-VL-Embedding | 공식 학습 코드 미공개 | 확인 불가 | 확인 불가 |
