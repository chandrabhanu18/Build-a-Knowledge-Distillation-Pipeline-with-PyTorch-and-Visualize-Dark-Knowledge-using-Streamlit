# Knowledge Distillation Pipeline with PyTorch and Streamlit

This project trains a compact teacher model, compresses it into a smaller student model with knowledge distillation, and exposes the results in a Streamlit dashboard.

## What is included

- A reproducible Docker + Docker Compose setup.
- A teacher training pipeline that saves a checkpoint, metrics, and precomputed logits.
- A student distillation sweep over temperature and alpha.
- A Streamlit app with:
  - Soft Target Inspector
  - Dark Knowledge Viewer
  - Compression Dashboard
  - Distillation Curve
- A `submission.json` file that points evaluators to the generated artifacts.

## Quick start

1. Build and run the stack:

```bash
docker-compose up --build -d
```

2. Open the app at:

```text
http://localhost:8501
```

## Generated artifacts

- `output/teacher/teacher_model.pth`
- `output/teacher/teacher_metrics.json`
- `output/teacher/teacher_train_logits.pt`
- `output/teacher/teacher_test_logits.pt`
- `output/student/best_student_model.pth`
- `output/student/best_student_metrics.json`
- `output/sweep_results.csv`
- `output/student/distillation_histories.json`

## Design notes

The default dataset is a fast, deterministic CIFAR-10-compatible synthetic dataset. It keeps the full pipeline runnable in constrained environments and still produces a strong teacher, a compressed student, and meaningful dark-knowledge examples. The code also keeps the conventional CIFAR-10 class names and can be extended to a real dataset if desired.

## Useful commands

Train only the teacher:

```bash
python -m src.train_teacher
```

Run the full pipeline:

```bash
python -m src.pipeline
```

Run the Streamlit app locally:

```bash
streamlit run src/app.py
```

## Submission checklist

- **Start Docker:** Ensure Docker Desktop / Docker Engine is running on the host.
- **Build & run stack (preferred):**
```powershell
$env:DOCKER_CONFIG = Join-Path $env:TEMP 'docker-no-creds'
docker compose up --build -d
```
- **Run Playwright checks (container):**
```powershell
$env:DOCKER_CONFIG = Join-Path $env:TEMP 'docker-no-creds'
docker compose build playwright
docker compose run --rm playwright
```
- **Run Playwright checks (local dev, no Docker):**
```powershell
pip install playwright
python -m playwright install --with-deps chromium
python tests/playwright_check.py
```
- **Validate artifacts (always run before submit):**
```powershell
python tests/validate_submission.py
```

Notes:
- The Playwright package and browser binaries are installed inside the `playwright` test image used by the CI checks; if you run tests locally, install Playwright and the browsers as shown above.
- The `submission.json` references files under `output/` — ensure those artifacts are present and up-to-date before submitting.
