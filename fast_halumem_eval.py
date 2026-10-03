import os
import copy
import random
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from tqdm import tqdm


# ============================================================
# Reproducibility
# ============================================================
from torchvision.models import vit_b_16, ViT_B_16_Weights

def build_vit(num_classes=10):
    weights = ViT_B_16_Weights.IMAGENET1K_V1

    model = vit_b_16(weights=weights)

    model.heads.head = nn.Linear(
        model.heads.head.in_features,
        num_classes
    )

    clean_transform, aug_transform = build_cnn_transforms()

    return model, clean_transform, aug_transform
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# Mixed transform
#
# original_only:
#     clean(img)
#
# augmented_only:
#     augmentation(img)
#
# mixed:
#     P=0.5 -> clean(img)
#     P=0.5 -> augmentation(img)
# ============================================================

class MixedTransform:
    def __init__(
        self,
        clean_transform: Callable,
        aug_transform: Callable,
        p_aug: float = 0.5,
    ):
        self.clean_transform = clean_transform
        self.aug_transform = aug_transform
        self.p_aug = p_aug

    def __call__(self, image):
        if torch.rand(1).item() < self.p_aug:
            return self.aug_transform(image)

        return self.clean_transform(image)


# ============================================================
# Robustness corruptions
# ============================================================

class RotateFixed:
    def __init__(self, degree=20):
        self.degree = degree

    def __call__(self, image):
        return image.rotate(
            self.degree,
            resample=Image.Resampling.BILINEAR,
        )


class CenterOcclusion:
    """
    가운데 부분을 가려서 occlusion robustness 측정.
    """

    def __init__(self, ratio=0.30):
        self.ratio = ratio

    def __call__(self, image):
        image = image.copy()

        w, h = image.size

        box_w = int(w * self.ratio)
        box_h = int(h * self.ratio)

        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [x1, y1, x2, y2],
            fill=(0, 0, 0),
        )

        return image


class LowContrast:
    def __init__(self, factor=0.45):
        self.factor = factor

    def __call__(self, image):
        return ImageEnhance.Contrast(image).enhance(
            self.factor
        )


class Darken:
    def __init__(self, factor=0.50):
        self.factor = factor

    def __call__(self, image):
        return ImageEnhance.Brightness(image).enhance(
            self.factor
        )


# ============================================================
# CNN transforms
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_cnn_transforms():

    # --------------------------------------------------------
    # "원본"
    #
    # Resize / Normalize은 augmentation이 아니라
    # network input preprocessing으로 취급
    # --------------------------------------------------------

    clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])

    # --------------------------------------------------------
    # augmentation
    #
    # RandomResizedCrop은 crop + resize를 수행.
    # --------------------------------------------------------

    augment = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.70, 1.0),
            ratio=(0.85, 1.15),
            antialias=True,
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.20,
                hue=0.05,
            )
        ], p=0.8),

        transforms.RandomRotation(
            degrees=15,
        ),

        transforms.ToTensor(),

        transforms.RandomErasing(
            p=0.20,
            scale=(0.02, 0.12),
            ratio=(0.5, 2.0),
        ),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])

    return clean, augment
# ============================================================
# CNN
# ============================================================

def build_cnn(num_classes=10):

    weights = ResNet18_Weights.IMAGENET1K_V1

    model = resnet18(
        weights=weights,
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    clean_transform, aug_transform = build_cnn_transforms()

    return (
        model,
        clean_transform,
        aug_transform,
    )


# ============================================================
# VLM / CLIP classifier
# ============================================================

class CLIPClassifier(nn.Module):

    def __init__(
        self,
        clip_model,
        text_features,
    ):
        super().__init__()

        self.clip = clip_model

        self.register_buffer(
            "text_features",
            text_features,
        )

    def forward(self, images):

        image_features = self.clip.encode_image(images)

        image_features = F.normalize(
            image_features,
            dim=-1,
        )

        # CLIP similarity scale
        scale = self.clip.logit_scale.exp()

        # 너무 커지는 것 방지
        scale = torch.clamp(scale, max=100)

        logits = (
            scale
            * image_features
            @ self.text_features.T
        )

        return logits


def build_vlm(
    device,
    class_names,
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
):

    import open_clip

    # OpenCLIP에서 training / validation transform을
    # 각각 제공한다.
    clip_model, preprocess_train, preprocess_val = \
        open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

    clip_model = clip_model.to(device)

    tokenizer = open_clip.get_tokenizer(
        model_name
    )

    # --------------------------------------------------------
    # Text prototypes
    # --------------------------------------------------------

    prompts = [
        f"a photo of a {name.replace('_', ' ')}"
        for name in class_names
    ]

    tokens = tokenizer(prompts).to(device)

    clip_model.eval()

    with torch.no_grad():

        text_features = clip_model.encode_text(
            tokens
        )

        text_features = F.normalize(
            text_features,
            dim=-1,
        )

    # --------------------------------------------------------
    # text tower freeze
    # visual encoder fine-tuning
    # --------------------------------------------------------

    for param in clip_model.parameters():
        param.requires_grad = False

    for param in clip_model.visual.parameters():
        param.requires_grad = True

    # logit scale도 학습
    clip_model.logit_scale.requires_grad = True

    classifier = CLIPClassifier(
        clip_model,
        text_features,
    )

    return (
        classifier,
        preprocess_val,       # clean
        preprocess_train,     # augmentation
    )


# ============================================================
# Dataset creation
# ============================================================

def get_split_indices(
    root,
    val_size=5000,
    split_seed=12345,
):

    dataset = CIFAR10(
        root=root,
        train=True,
        download=False,
    )

    generator = torch.Generator()
    generator.manual_seed(split_seed)

    indices = torch.randperm(
        len(dataset),
        generator=generator,
    ).tolist()

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return train_indices, val_indices


def create_train_dataset(
    root,
    indices,
    transform,
):

    dataset = CIFAR10(
        root=root,
        train=True,
        transform=transform,
        download=False,
    )

    return Subset(
        dataset,
        indices,
    )


def create_val_dataset(
    root,
    indices,
    transform,
):

    dataset = CIFAR10(
        root=root,
        train=True,
        transform=transform,
        download=False,
    )

    return Subset(
        dataset,
        indices,
    )


def create_test_dataset(
    root,
    base_transform,
    corruption=None,
):

    if corruption is None:
        transform = base_transform

    else:
        transform = transforms.Compose([
            corruption,
            base_transform,
        ])

    return CIFAR10(
        root=root,
        train=False,
        transform=transform,
        download=False,
    )


# ============================================================
# Loader
# ============================================================

def build_loader(
    dataset,
    batch_size,
    shuffle,
    workers,
    seed,
):

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


# ============================================================
# train
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total = 0

    use_amp = device.type == "cuda"

    pbar = tqdm(
        loader,
        leave=False,
    )

    for images, targets in pbar:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(images)

            loss = F.cross_entropy(
                logits,
                targets,
            )

        scaler.scale(loss).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == targets
        ).sum().item()

        total += targets.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
    }


# ============================================================
# evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    total_correct = 0
    total = 0

    use_amp = device.type == "cuda"

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(
                images
            )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == targets
        ).sum().item()

        total += targets.size(0)

    return total_correct / total


# ============================================================
# Single experiment
# ============================================================

def run_experiment(
    args,
    regime,
    seed,
    train_indices,
    val_indices,
    class_names,
):

    seed_everything(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print(
        f"model={args.model} "
        f"regime={regime} "
        f"seed={seed}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # fresh initialization
    # 매우 중요:
    # regime마다 같은 pretrained checkpoint에서 다시 시작
    # --------------------------------------------------------

    if args.model == "cnn":

        model, clean_transform, aug_transform = \
            build_cnn(
                num_classes=len(class_names)
            )

        model = model.to(device)

        lr = args.cnn_lr

    else:

        model, clean_transform, aug_transform = \
            build_vlm(
                device=device,
                class_names=class_names,
                model_name=args.vlm_model,
                pretrained=args.vlm_pretrained,
            )

        model = model.to(device)

        lr = args.vlm_lr

    # --------------------------------------------------------
    # regime
    # --------------------------------------------------------

    if regime == "original_only":

        train_transform = clean_transform

    elif regime == "augmented_only":

        train_transform = aug_transform

    elif regime == "mixed":

        train_transform = MixedTransform(
            clean_transform=clean_transform,
            aug_transform=aug_transform,
            p_aug=args.mixed_p,
        )

    else:
        raise ValueError(
            regime
        )

    # --------------------------------------------------------
    # datasets
    # --------------------------------------------------------

    train_dataset = create_train_dataset(
        args.data_dir,
        train_indices,
        train_transform,
    )

    val_dataset = create_val_dataset(
        args.data_dir,
        val_indices,
        clean_transform,
    )

    clean_test = create_test_dataset(
        args.data_dir,
        clean_transform,
    )

    rotate_test = create_test_dataset(
        args.data_dir,
        clean_transform,
        corruption=RotateFixed(
            args.rotation_degree
        ),
    )

    occlusion_test = create_test_dataset(
        args.data_dir,
        clean_transform,
        corruption=CenterOcclusion(
            args.occlusion_ratio
        ),
    )

    contrast_test = create_test_dataset(
        args.data_dir,
        clean_transform,
        corruption=LowContrast(
            0.45
        ),
    )

    dark_test = create_test_dataset(
        args.data_dir,
        clean_transform,
        corruption=Darken(
            0.50
        ),
    )

    # --------------------------------------------------------
    # loaders
    # --------------------------------------------------------

    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        True,
        args.workers,
        seed,
    )

    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        False,
        args.workers,
        seed,
    )

    test_loaders = {

        "clean":
            build_loader(
                clean_test,
                args.batch_size,
                False,
                args.workers,
                seed,
            ),

        "rotate":
            build_loader(
                rotate_test,
                args.batch_size,
                False,
                args.workers,
                seed,
            ),

        "occlusion":
            build_loader(
                occlusion_test,
                args.batch_size,
                False,
                args.workers,
                seed,
            ),

        "low_contrast":
            build_loader(
                contrast_test,
                args.batch_size,
                False,
                args.workers,
                seed,
            ),

        "dark":
            build_loader(
                dark_test,
                args.batch_size,
                False,
                args.workers,
                seed,
            ),
    }

    # --------------------------------------------------------
    # optimizer
    # --------------------------------------------------------

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    # --------------------------------------------------------
    # training
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1
    ):

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
        )

        val_acc = evaluate(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_acc:.4f}"
        )

    # --------------------------------------------------------
    # final test
    # --------------------------------------------------------

    metrics = {}

    for name, loader in test_loaders.items():

        acc = evaluate(
            model,
            loader,
            device,
        )

        metrics[name] = acc

    robust_values = [
        metrics["rotate"],
        metrics["occlusion"],
        metrics["low_contrast"],
        metrics["dark"],
    ]

    metrics["robust_avg"] = float(
        np.mean(robust_values)
    )

    # clean 대비 degradation
    metrics["robust_drop"] = (
        metrics["clean"]
        - metrics["robust_avg"]
    )

    print()
    print("Final results")

    for k, v in metrics.items():
        print(
            f"{k:15s}: {v:.4f}"
        )

    # memory cleanup

    del model
    del optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["cnn", "vlm", "vit"],
        default="cnn",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--mixed-p",
        type=float,
        default=0.5,
        help="mixed에서 augmentation 선택 확률",
    )

    parser.add_argument(
        "--cnn-lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--vlm-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--rotation-degree",
        type=float,
        default=20,
    )

    parser.add_argument(
        "--occlusion-ratio",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--val-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
    )

    # OpenCLIP

    parser.add_argument(
        "--vlm-model",
        type=str,
        default="ViT-B-32",
    )

    parser.add_argument(
        "--vlm-pretrained",
        type=str,
        default="laion2b_s34b_b79k",
    )

    args = parser.parse_args()

    os.makedirs(
        args.data_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Download CIFAR-10 once
    # --------------------------------------------------------

    print("Downloading/checking CIFAR-10...")

    train_base = CIFAR10(
        args.data_dir,
        train=True,
        download=True,
    )

    CIFAR10(
        args.data_dir,
        train=False,
        download=True,
    )

    class_names = train_base.classes

    del train_base

    train_indices, val_indices = \
        get_split_indices(
            args.data_dir,
            val_size=args.val_size,
        )

    regimes = [
        "original_only",
        "augmented_only",
        "mixed",
    ]

    all_rows = []

    # --------------------------------------------------------
    # experiments
    # --------------------------------------------------------

    for seed in args.seeds:

        for regime in regimes:

            metrics = run_experiment(
                args=args,
                regime=regime,
                seed=seed,
                train_indices=train_indices,
                val_indices=val_indices,
                class_names=class_names,
            )

            row = {
                "model": args.model,
                "seed": seed,
                "regime": regime,
            }

            row.update(metrics)

            all_rows.append(row)

    # ========================================================
    # Results
    # ========================================================

    df = pd.DataFrame(
        all_rows
    )

    print()
    print("=" * 100)
    print("RAW RESULTS")
    print("=" * 100)

    print(
        df.to_string(
            index=False
        )
    )

    df.to_csv(
        f"augmentation_results_{args.model}.csv",
        index=False,
    )

    # --------------------------------------------------------
    # seed 평균
    # --------------------------------------------------------

    metric_columns = [
        "clean",
        "rotate",
        "occlusion",
        "low_contrast",
        "dark",
        "robust_avg",
        "robust_drop",
    ]

    summary_mean = (
        df
        .groupby("regime")[metric_columns]
        .mean()
    )

    summary_std = (
        df
        .groupby("regime")[metric_columns]
        .std()
    )

    print()
    print("=" * 100)
    print("MEAN")
    print("=" * 100)

    print(
        summary_mean.round(4)
    )

    print()
    print("=" * 100)
    print("STD")
    print("=" * 100)

    print(
        summary_std.round(4)
    )

    summary_mean.to_csv(
        f"augmentation_summary_mean_{args.model}.csv"
    )

    summary_std.to_csv(
        f"augmentation_summary_std_{args.model}.csv"
    )


if __name__ == "__main__":
    main()
