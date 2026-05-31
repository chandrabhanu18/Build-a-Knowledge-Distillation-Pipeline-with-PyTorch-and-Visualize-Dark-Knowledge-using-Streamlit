from __future__ import annotations

import argparse

from .config import RANDOM_SEED, SWEEP_EPOCHS, TEACHER_EPOCHS, TRAIN_BATCH_SIZE, ensure_project_dirs
from .distill import run_distillation_sweep
from .train_teacher import train_teacher_model


def run_pipeline(force: bool = False) -> None:
    ensure_project_dirs()
    train_teacher_model(
        epochs=TEACHER_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        lr=0.001,
        weight_decay=0.0001,
        seed=RANDOM_SEED,
        force=force,
    )
    run_distillation_sweep(
        epochs=SWEEP_EPOCHS,
        lr=0.001,
        weight_decay=0.0001,
        seed=RANDOM_SEED,
        force=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full knowledge distillation pipeline")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_pipeline(force=args.force)


if __name__ == "__main__":
    main()
