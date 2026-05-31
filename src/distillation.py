from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .config import DEVICE, OUTPUT_DIR
from .models import build_student_model
from .utils import count_parameters, ensure_dir, save_json


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, true_labels: torch.Tensor, temperature: float, alpha: float) -> torch.Tensor:
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    kl_div_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean") * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, true_labels)
    return alpha * ce_loss + (1.0 - alpha) * kl_div_loss


@dataclass
class EvaluationResult:
    accuracy: float
    loss: float


def evaluate(model: nn.Module, data_loader, device: str = DEVICE) -> EvaluationResult:
    model = model.to(device)
    model.eval()
    total_correct = 0
    total_examples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)
    return EvaluationResult(accuracy=total_correct / max(total_examples, 1), loss=total_loss / max(total_examples, 1))


def train_student(
    train_loader,
    test_loader,
    teacher_logits: torch.Tensor,
    temperature: float,
    alpha: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str = DEVICE,
    seed: int = 42,
    student_model: nn.Module | None = None,
) -> tuple[nn.Module, dict[str, list[float]], EvaluationResult]:
    torch.manual_seed(seed)
    model = (student_model or build_student_model()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"epoch": [], "train_loss": [], "test_accuracy": []}

    teacher_logits = teacher_logits.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels, indices in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            indices = indices.to(device)
            batch_teacher_logits = teacher_logits[indices]
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(images)
            loss = distillation_loss(student_logits, batch_teacher_logits, labels, temperature, alpha)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        eval_result = evaluate(model, test_loader, device=device)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(running_loss / max(seen, 1))
        history["test_accuracy"].append(eval_result.accuracy)

    final_eval = evaluate(model, test_loader, device=device)
    return model, history, final_eval


def save_student_artifacts(model: nn.Module, checkpoint_path: Path, metrics_path: Path, accuracy: float, best_temp: float, best_alpha: float) -> dict[str, object]:
    ensure_dir(checkpoint_path.parent)
    torch.save(model.state_dict(), checkpoint_path)
    payload = {
        "accuracy": float(accuracy),
        "params": int(count_parameters(model)),
        "best_temp": int(best_temp),
        "best_alpha": float(best_alpha),
    }
    save_json(metrics_path, payload)
    return payload
