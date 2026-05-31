from __future__ import annotations

import base64
import json
import random
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .config import ROOT_DIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def state_dict_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().clamp(0, 1)
    if image.ndim != 3:
        raise ValueError("Expected a CHW tensor")
    array = (image.mul(255).byte().permute(1, 2, 0).numpy())
    return Image.fromarray(array)


def tensor_to_base64(image_tensor: torch.Tensor) -> str:
    buffer = BytesIO()
    tensor_to_pil(image_tensor).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def benchmark_inference_ms(model: torch.nn.Module, sample: torch.Tensor, runs: int = 25) -> float:
    model = model.eval()
    sample = sample.unsqueeze(0)
    with torch.no_grad():
        for _ in range(3):
            model(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(runs):
            model(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = perf_counter() - start
    return (elapsed / runs) * 1000.0


def softmax_numpy(logits: Iterable[float], temperature: float = 1.0) -> list[float]:
    values = np.asarray(list(logits), dtype=np.float64) / max(temperature, 1e-6)
    values = values - values.max()
    probs = np.exp(values)
    probs = probs / probs.sum()
    return probs.tolist()


def project_path(*parts: str) -> Path:
    return ROOT_DIR.joinpath(*parts)
