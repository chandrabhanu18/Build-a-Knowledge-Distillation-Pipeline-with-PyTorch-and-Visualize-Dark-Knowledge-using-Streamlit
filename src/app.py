from __future__ import annotations

import torch
import streamlit as st
from streamlit.components.v1 import html as st_html

from src.config import CLASS_NAMES, OUTPUT_DIR
from src.data import build_datasets
from src.models import build_student_model, build_teacher_model
from src.utils import benchmark_inference_ms, load_json, state_dict_size_mb, tensor_to_base64
from src.visualization import (
    build_compression_dashboard_html,
    build_dark_knowledge_viewer_html,
    build_distillation_curve_html,
    build_soft_target_inspector_html,
)


st.set_page_config(page_title="Knowledge Distillation Dashboard", page_icon="KD", layout="wide")


@st.cache_resource
def load_teacher_model():
    model = build_teacher_model()
    checkpoint_path = OUTPUT_DIR / "teacher" / "teacher_model.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def load_assets():
    bundle = build_datasets()
    teacher_metrics = load_json(OUTPUT_DIR / "teacher" / "teacher_metrics.json")
    student_metrics = load_json(OUTPUT_DIR / "student" / "best_student_metrics.json")
    sweep_path = OUTPUT_DIR / "sweep_results.csv"
    sweep_df = None
    if sweep_path.exists():
        import pandas as pd

        sweep_df = pd.read_csv(sweep_path)
    histories = load_json(OUTPUT_DIR / "student" / "distillation_histories.json")
    teacher_logits_payload = torch.load(OUTPUT_DIR / "teacher" / "teacher_test_logits.pt", map_location="cpu") if (OUTPUT_DIR / "teacher" / "teacher_test_logits.pt").exists() else None
    return bundle, teacher_metrics, student_metrics, sweep_df, histories, teacher_logits_payload


def _predict_logits(model, image_tensor):
    with torch.no_grad():
        return model(image_tensor.unsqueeze(0)).squeeze(0).cpu().tolist()


def _tensor_from_upload(uploaded_file):
    from PIL import Image
    import numpy as np

    image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    array = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _build_dark_examples(bundle, teacher_logits_payload):
    test_dataset = bundle.test
    logits = teacher_logits_payload["logits"] if teacher_logits_payload is not None else None
    if logits is None:
        return []
    probs = torch.softmax(logits, dim=1)
    examples = []
    for index in range(min(len(test_dataset), len(probs))):
        image, label, _ = test_dataset[index]
        probability_vector = probs[index]
        pred_prob, pred_idx = probability_vector.max(dim=0)
        if int(pred_idx) != int(label):
            continue
        wrong_entries = []
        for class_index, class_name in enumerate(CLASS_NAMES):
            if class_index == int(label):
                continue
            value = float(probability_vector[class_index].item())
            if value > 0.05:
                wrong_entries.append({"label": class_name, "prob": value})
        if wrong_entries:
            wrong_entries.sort(key=lambda item: item["prob"], reverse=True)
            examples.append(
                {
                    "image_b64": tensor_to_base64(image),
                    "label_name": CLASS_NAMES[int(label)],
                    "pred_name": CLASS_NAMES[int(pred_idx)],
                    "pred_prob": float(pred_prob.item()),
                    "wrong_classes": wrong_entries[:3],
                }
            )
        if len(examples) >= 4:
            break
    return examples


def main() -> None:
    st.title("Knowledge Distillation Pipeline")
    st.caption("Teacher, student, dark knowledge, temperature scaling, and model compression in one place.")

    bundle, teacher_metrics, student_metrics, sweep_df, histories, teacher_logits_payload = load_assets()
    teacher_model = load_teacher_model()

    if not teacher_metrics or not student_metrics:
        st.error("Run the training pipeline first so the dashboard can load artifacts from output/.")
        return

    test_dataset = bundle.test
    sample_index = st.selectbox("Select a test sample", list(range(min(48, len(test_dataset)))), index=0)
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        sample_tensor = _tensor_from_upload(uploaded)
        sample_label = "uploaded image"
        sample_image_b64 = tensor_to_base64(sample_tensor)
        sample_logits = _predict_logits(teacher_model, sample_tensor)
    else:
        sample_tensor, sample_label_tensor, _ = test_dataset[sample_index]
        sample_label = CLASS_NAMES[int(sample_label_tensor)]
        sample_image_b64 = tensor_to_base64(sample_tensor)
        sample_logits = teacher_logits_payload["logits"][sample_index].tolist() if teacher_logits_payload is not None else _predict_logits(teacher_model, sample_tensor)

    teacher_checkpoint = OUTPUT_DIR / "teacher" / "teacher_model.pth"
    student_checkpoint = OUTPUT_DIR / "student" / "best_student_model.pth"

    teacher_dashboard = {
        "params": teacher_metrics["params"],
        "accuracy": teacher_metrics["accuracy"],
        "model_size_mb": state_dict_size_mb(teacher_checkpoint),
        "inference_ms": benchmark_inference_ms(teacher_model, sample_tensor),
    }
    student_dashboard = {
        "params": student_metrics["params"],
        "accuracy": student_metrics["accuracy"],
        "model_size_mb": state_dict_size_mb(student_checkpoint),
        "inference_ms": 0.0,
    }
    if student_checkpoint.exists():
        student_model = build_student_model()
        student_model.load_state_dict(torch.load(student_checkpoint, map_location="cpu"))
        student_dashboard["inference_ms"] = benchmark_inference_ms(student_model, sample_tensor)

    best_temp = int(student_metrics.get("best_temp", 4))
    soft_html = build_soft_target_inspector_html(sample_image_b64, sample_logits, CLASS_NAMES, f"Soft Target Inspector - {sample_label}", initial_temperature=float(best_temp))
    dark_examples = _build_dark_examples(bundle, teacher_logits_payload)
    if not dark_examples:
        dark_examples = [
            {
                "image_b64": sample_image_b64,
                "label_name": sample_label,
                "pred_name": sample_label,
                "pred_prob": 1.0,
                "wrong_classes": [],
            }
        ]

    st.markdown("### Interactive views")
    st_html(soft_html, height=620, scrolling=False)
    st_html(build_dark_knowledge_viewer_html(dark_examples), height=700, scrolling=True)
    st_html(build_compression_dashboard_html(teacher_dashboard, student_dashboard), height=280, scrolling=False)
    st_html(build_distillation_curve_html(histories), height=420, scrolling=False)


if __name__ == "__main__":
    main()
