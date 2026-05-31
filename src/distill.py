from __future__ import annotations

import argparse
import csv

import torch

from .config import DEVICE, OUTPUT_DIR, RANDOM_SEED, SWEEP_EPOCHS, TEST_BATCH_SIZE, TRAIN_BATCH_SIZE, ensure_project_dirs
from .data import build_dataloaders, build_datasets
from .distillation import save_student_artifacts, train_student
from .models import build_student_model, build_teacher_model
from .train_teacher import train_teacher_model
from .utils import count_parameters, ensure_dir, load_json, save_json, set_seed


def _load_teacher_logits() -> torch.Tensor:
    payload = torch.load(OUTPUT_DIR / "teacher" / "teacher_train_logits.pt", map_location="cpu")
    return payload["logits"].float()


def _load_teacher_model() -> torch.nn.Module:
    model = build_teacher_model()
    model.load_state_dict(torch.load(OUTPUT_DIR / "teacher" / "teacher_model.pth", map_location="cpu"))
    return model


def _evaluate_parameter_ratio() -> float:
    teacher_model = _load_teacher_model()
    student_model = build_student_model()
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    ratio = teacher_params / max(student_params, 1)
    if not (5.0 <= ratio <= 10.0):
        raise RuntimeError(f"Parameter ratio out of range: {ratio:.2f}x")
    return ratio


def run_distillation_sweep(epochs: int, lr: float, weight_decay: float, seed: int = RANDOM_SEED, force: bool = False):
    ensure_project_dirs()
    student_dir = ensure_dir(OUTPUT_DIR / "student")
    sweep_csv_path = OUTPUT_DIR / "sweep_results.csv"
    best_checkpoint_path = student_dir / "best_student_model.pth"
    best_metrics_path = student_dir / "best_student_metrics.json"
    histories_path = student_dir / "distillation_histories.json"

    if sweep_csv_path.exists() and best_checkpoint_path.exists() and best_metrics_path.exists() and histories_path.exists() and not force:
        return load_json(best_metrics_path)

    set_seed(seed)
    bundle = build_datasets()
    train_loader, test_loader = build_dataloaders(bundle, train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE)
    teacher_logits = _load_teacher_logits()
    _evaluate_parameter_ratio()

    temperatures = [1, 2, 4, 8, 16]
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    rows = []
    best_row = None
    best_state_dict = None
    best_history = None
    best_model = None

    for temperature in temperatures:
        for alpha in alphas:
            model, history, evaluation = train_student(
                train_loader=train_loader,
                test_loader=test_loader,
                teacher_logits=teacher_logits,
                temperature=float(temperature),
                alpha=float(alpha),
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                device=DEVICE,
                seed=seed + int(temperature * 100 + alpha * 10),
            )
            rows.append({"temperature": temperature, "alpha": alpha, "student_accuracy": float(evaluation.accuracy)})
            if best_row is None or evaluation.accuracy > best_row["student_accuracy"]:
                best_row = {"temperature": temperature, "alpha": alpha, "student_accuracy": float(evaluation.accuracy)}
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_history = history
                best_model = model

    with sweep_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["temperature", "alpha", "student_accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    if best_model is None or best_row is None or best_history is None:
        raise RuntimeError("Distillation sweep did not produce a best model")

    best_model.load_state_dict(best_state_dict)
    save_student_artifacts(
        model=best_model,
        checkpoint_path=best_checkpoint_path,
        metrics_path=best_metrics_path,
        accuracy=best_row["student_accuracy"],
        best_temp=best_row["temperature"],
        best_alpha=best_row["alpha"],
    )

    selected_temp = int(best_row["temperature"])
    histories = {}
    for label, alpha in [("hard_only", 1.0), ("soft_only", 0.0), ("best_alpha", float(best_row["alpha"]))]:
        _model, history, _ = train_student(
            train_loader=train_loader,
            test_loader=test_loader,
            teacher_logits=teacher_logits,
            temperature=float(selected_temp),
            alpha=float(alpha),
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=DEVICE,
            seed=seed + 999 + int(alpha * 100),
        )
        histories[label] = {"temperature": selected_temp, "alpha": float(alpha), "history": history}

    save_json(histories_path, histories)
    return load_json(best_metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run knowledge distillation sweep")
    parser.add_argument("--epochs", type=int, default=SWEEP_EPOCHS)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    train_teacher_model(epochs=8, batch_size=TRAIN_BATCH_SIZE, lr=0.001, weight_decay=0.0001, seed=args.seed, force=False)
    run_distillation_sweep(args.epochs, args.lr, args.weight_decay, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
