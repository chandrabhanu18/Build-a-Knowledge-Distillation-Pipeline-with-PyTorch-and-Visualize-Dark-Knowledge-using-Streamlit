from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from .config import DEVICE, OUTPUT_DIR, RANDOM_SEED, TEACHER_EPOCHS, TEST_BATCH_SIZE, TRAIN_BATCH_SIZE, ensure_project_dirs
from .data import build_dataloaders, build_datasets
from .distillation import evaluate
from .models import build_teacher_model
from .utils import count_parameters, ensure_dir, save_json, set_seed


def compute_logits(model, data_loader, device: str = DEVICE):
    model = model.to(device)
    model.eval()
    all_logits = []
    all_labels = []
    all_indices = []
    with torch.no_grad():
        for images, labels, indices in data_loader:
            images = images.to(device)
            logits = model(images).cpu()
            all_logits.append(logits)
            all_labels.append(labels)
            all_indices.append(indices)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0), torch.cat(all_indices, dim=0)


def train_teacher_model(epochs: int, batch_size: int, lr: float, weight_decay: float, seed: int = RANDOM_SEED, force: bool = False):
    ensure_project_dirs()
    teacher_dir = ensure_dir(OUTPUT_DIR / "teacher")
    checkpoint_path = teacher_dir / "teacher_model.pth"
    metrics_path = teacher_dir / "teacher_metrics.json"
    train_logits_path = teacher_dir / "teacher_train_logits.pt"
    test_logits_path = teacher_dir / "teacher_test_logits.pt"

    if checkpoint_path.exists() and metrics_path.exists() and train_logits_path.exists() and test_logits_path.exists() and not force:
        return torch.load(checkpoint_path, map_location="cpu")

    set_seed(seed)
    bundle = build_datasets()
    train_loader, test_loader = build_dataloaders(bundle, train_batch_size=batch_size, test_batch_size=TEST_BATCH_SIZE)
    model = build_teacher_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = torch.nn.CrossEntropyLoss()

    best_state = None
    best_accuracy = -1.0

    for _epoch in range(epochs):
        model.train()
        for images, labels, _indices in tqdm(train_loader, desc="Training teacher", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        test_result = evaluate(model, test_loader)
        if test_result.accuracy > best_accuracy:
            best_accuracy = test_result.accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_accuracy = evaluate(model, train_loader).accuracy
    test_accuracy = evaluate(model, test_loader).accuracy
    params = count_parameters(model)
    torch.save(model.state_dict(), checkpoint_path)
    save_json(metrics_path, {"accuracy": float(test_accuracy), "params": int(params), "train_accuracy": float(train_accuracy)})

    train_logits, train_labels, train_indices = compute_logits(model, train_loader)
    test_logits, test_labels, test_indices = compute_logits(model, test_loader)
    torch.save({"logits": train_logits, "labels": train_labels, "indices": train_indices}, train_logits_path)
    torch.save({"logits": test_logits, "labels": test_labels, "indices": test_indices}, test_logits_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the teacher model")
    parser.add_argument("--epochs", type=int, default=TEACHER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    train_teacher_model(args.epochs, args.batch_size, args.lr, args.weight_decay, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
