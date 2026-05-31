from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from .config import CLASS_NAMES, DATASET_MODE, NUM_WORKERS, TEST_BATCH_SIZE, TRAIN_BATCH_SIZE


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    test: Dataset
    class_names: list[str]


def _class_palette(label: int) -> tuple[tuple[int, int, int], tuple[int, int, int], str]:
    palettes = [
        ((210, 65, 60), (255, 225, 225), "vertical"),
        ((195, 80, 70), (255, 230, 230), "horizontal"),
        ((70, 170, 90), (225, 255, 225), "circle"),
        ((85, 160, 95), (230, 255, 230), "cross"),
        ((75, 120, 205), (225, 235, 255), "diagonal"),
        ((85, 130, 195), (230, 238, 255), "anti_diagonal"),
        ((220, 190, 70), (255, 250, 220), "border"),
        ((205, 180, 80), (255, 248, 225), "corners"),
        ((150, 90, 210), (245, 230, 255), "x"),
        ((160, 100, 205), (240, 225, 255), "dots"),
    ]
    return palettes[label]


def _draw_pattern(draw: ImageDraw.ImageDraw, size: int, pattern: str) -> None:
    margin = size // 6
    inner = size - margin
    width = max(1, size // 8)
    if pattern == "vertical":
        draw.rectangle([size // 2 - width // 2, margin, size // 2 + width // 2, inner], fill=(255, 255, 255))
    elif pattern == "horizontal":
        draw.rectangle([margin, size // 2 - width // 2, inner, size // 2 + width // 2], fill=(255, 255, 255))
    elif pattern == "circle":
        draw.ellipse([margin, margin, inner, inner], outline=(255, 255, 255), width=width)
    elif pattern == "cross":
        draw.line([margin, margin, inner, inner], fill=(255, 255, 255), width=width)
        draw.line([margin, inner, inner, margin], fill=(255, 255, 255), width=width)
    elif pattern == "diagonal":
        draw.line([margin, inner, inner, margin], fill=(255, 255, 255), width=width)
    elif pattern == "anti_diagonal":
        draw.line([margin, margin, inner, inner], fill=(255, 255, 255), width=width)
    elif pattern == "border":
        draw.rectangle([margin, margin, inner, inner], outline=(255, 255, 255), width=width)
    elif pattern == "corners":
        corner = size // 8
        draw.rectangle([margin, margin, margin + corner, margin + corner], fill=(255, 255, 255))
        draw.rectangle([inner - corner, margin, inner, margin + corner], fill=(255, 255, 255))
        draw.rectangle([margin, inner - corner, margin + corner, inner], fill=(255, 255, 255))
        draw.rectangle([inner - corner, inner - corner, inner, inner], fill=(255, 255, 255))
    elif pattern == "x":
        draw.line([margin, size // 2, size // 2, margin], fill=(255, 255, 255), width=width)
        draw.line([size // 2, margin, inner, size // 2], fill=(255, 255, 255), width=width)
        draw.line([size // 2, inner, inner, size // 2], fill=(255, 255, 255), width=width)
        draw.line([margin, size // 2, size // 2, inner], fill=(255, 255, 255), width=width)
    elif pattern == "dots":
        radius = size // 12
        centers = [(size // 2, size // 2), (size // 3, size // 3), (2 * size // 3, size // 3), (size // 3, 2 * size // 3), (2 * size // 3, 2 * size // 3)]
        for cx, cy in centers:
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 255, 255))


def generate_synthetic_image(label: int, index: int, split: str, size: int = 32) -> torch.Tensor:
    background, accent, pattern = _class_palette(label)
    seed = 10_000 + label * 1_000 + index + (0 if split == "train" else 50_000)
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)

    if rng.random() > 0.5:
        inset = size // 8
        draw.rectangle([inset, inset, size - inset, size - inset], fill=tuple(int(v) for v in accent))
    _draw_pattern(draw, size, pattern)

    for _ in range(rng.integers(1, 4)):
        x0 = int(rng.integers(0, size - 6))
        y0 = int(rng.integers(0, size - 6))
        x1 = x0 + int(rng.integers(3, 8))
        y1 = y0 + int(rng.integers(3, 8))
        color = tuple(int(v) for v in np.clip(np.array(accent) + rng.integers(-35, 35, size=3), 0, 255))
        draw.rectangle([x0, y0, x1, y1], outline=color)

    array = np.asarray(image).astype(np.float32) / 255.0
    noise = rng.normal(0.0, 0.045 if split == "train" else 0.03, size=array.shape).astype(np.float32)
    array = np.clip(array + noise, 0.0, 1.0)
    shift_x = int(rng.integers(-2, 3))
    shift_y = int(rng.integers(-2, 3))
    array = np.roll(array, shift=(shift_x, shift_y), axis=(0, 1))
    return torch.from_numpy(array).permute(2, 0, 1)


class SyntheticCIFAR10Dataset(Dataset):
    def __init__(self, split: str, samples_per_class: int):
        self.split = split
        self.samples_per_class = samples_per_class
        self.length = samples_per_class * len(CLASS_NAMES)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        label = index // self.samples_per_class
        within_class = index % self.samples_per_class
        image = generate_synthetic_image(label, within_class, self.split)
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(index, dtype=torch.long)


def build_datasets(train_samples_per_class: int = 120, test_samples_per_class: int = 30) -> DatasetBundle:
    train = SyntheticCIFAR10Dataset("train", train_samples_per_class)
    test = SyntheticCIFAR10Dataset("test", test_samples_per_class)
    return DatasetBundle(train=train, test=test, class_names=CLASS_NAMES)


def build_dataloaders(bundle: DatasetBundle, train_batch_size: int = TRAIN_BATCH_SIZE, test_batch_size: int = TEST_BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(bundle.train, batch_size=train_batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    test_loader = DataLoader(bundle.test, batch_size=test_batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    return train_loader, test_loader


def dataset_summary(bundle: DatasetBundle) -> dict[str, object]:
    return {
        "mode": DATASET_MODE,
        "train_size": len(bundle.train),
        "test_size": len(bundle.test),
        "class_names": bundle.class_names,
    }
