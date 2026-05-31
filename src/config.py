from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT_DIR / "output"))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
USE_CUDA = os.getenv("USE_CUDA", "0") == "1"
DATASET_MODE = os.getenv("DATASET_MODE", "synthetic").lower()

TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "64"))
TEST_BATCH_SIZE = int(os.getenv("TEST_BATCH_SIZE", "128"))
TEACHER_EPOCHS = int(os.getenv("TEACHER_EPOCHS", "8"))
STUDENT_EPOCHS = int(os.getenv("STUDENT_EPOCHS", "5"))
SWEEP_EPOCHS = int(os.getenv("SWEEP_EPOCHS", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0001"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))

DEVICE = "cuda" if USE_CUDA else "cpu"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = TEACHER_EPOCHS
    batch_size: int = TRAIN_BATCH_SIZE
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY


def ensure_project_dirs() -> None:
    for path in [OUTPUT_DIR, DATA_DIR, OUTPUT_DIR / "teacher", OUTPUT_DIR / "student"]:
        path.mkdir(parents=True, exist_ok=True)
