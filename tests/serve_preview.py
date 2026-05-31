import base64
import sys
from pathlib import Path

# ensure project root is on sys.path so `import src.*` works when running from tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import threading

from src.config import CLASS_NAMES


def load_json(path, default=None):
    try:
        p = Path(path)
        if not p.exists():
            return {} if default is None else default
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default

# The original preview imported builders from `src.visualization` which in turn
# imports `torch`. Importing torch on machines without a proper PyTorch
# installation (or where DLLs fail to load) breaks this lightweight preview.
# To keep the preview runnable without PyTorch, include lightweight copies of
# the HTML builder functions here that avoid importing torch.
import html
import json
import numpy as np


def _escape(text: str) -> str:
        return html.escape(text, quote=True)


def softmax_numpy(logits, temperature: float = 1.0):
        values = np.asarray(list(logits), dtype=np.float64) / max(temperature, 1e-6)
        values = values - values.max()
        probs = np.exp(values)
        probs = probs / probs.sum()
        return probs.tolist()


def build_soft_target_inspector_html(image_b64: str, logits: list[float], class_names: list[str], title: str, initial_temperature: float = 4.0) -> str:
        probs = softmax_numpy(logits, temperature=initial_temperature)
        bars = []
        for label, prob in sorted(zip(class_names, probs), key=lambda item: item[1], reverse=True):
                bars.append(
                        f'<div class="bar-row"><span class="bar-label">{_escape(label)}</span><div class="bar-track"><div class="bar-fill" style="width:{prob * 100:.2f}%"></div></div><span class="bar-value">{prob:.3f}</span></div>'
                )
        bars_html = "\n".join(bars)
        logits_js = ",".join(f"{float(value):.8f}" for value in logits)
        labels_js = ",".join(f'"{_escape(label)}"' for label in class_names)
        return f"""
<style>
    .kd-card {{ border: 1px solid #dbe4f0; border-radius: 18px; padding: 20px; background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); box-shadow: 0 14px 40px rgba(15, 23, 42, 0.08); }}
    .kd-header {{ display: flex; gap: 24px; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 18px; }}
    .kd-header h3, .kd-card h3 {{ margin: 0; color: #0f172a; }}
    .kd-header p, .kd-card p {{ margin: 6px 0 0 0; color: #475569; }}
    .temp-wrap {{ min-width: 240px; }}
    .temp-wrap label {{ display: block; margin-bottom: 8px; font-weight: 600; color: #0f172a; }}
    .temp-wrap input[type=range] {{ width: 100%; accent-color: #0f766e; }}
    .kd-body {{ display: grid; grid-template-columns: 160px 1fr; gap: 24px; align-items: start; }}
    .preview {{ width: 160px; height: 160px; object-fit: cover; border-radius: 16px; border: 1px solid #cbd5e1; background: white; }}
    .chart {{ display: grid; gap: 8px; }}
    .bar-row {{ display: grid; grid-template-columns: 92px 1fr 52px; gap: 10px; align-items: center; }}
    .bar-label {{ font-size: 0.88rem; color: #0f172a; }}
    .bar-track {{ height: 12px; border-radius: 999px; background: #e2e8f0; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: inherit; background: linear-gradient(90deg, #0f766e 0%, #38bdf8 100%); }}
    .bar-value {{ font-variant-numeric: tabular-nums; text-align: right; color: #0f172a; }}
    .examples-grid {{ display: grid; gap: 16px; }}
    .example-card {{ display: grid; grid-template-columns: 120px 1fr; gap: 16px; padding: 14px; border: 1px solid #cbd5e1; border-radius: 14px; background: white; }}
    .example-card img {{ width: 120px; height: 120px; object-fit: cover; border-radius: 12px; }}
    .compare-table {{ width: 100%; border-collapse: collapse; }}
    .compare-table th, .compare-table td {{ padding: 12px 14px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
    .compare-table thead th {{ background: #f8fafc; color: #0f172a; }}
</style>
<div data-testid="soft-target-inspector" class="kd-card">
    <div class="kd-header">
        <div>
            <h3>{_escape(title)}</h3>
            <p>Drag the temperature slider to soften or sharpen the teacher distribution.</p>
        </div>
        <div class="temp-wrap">
            <label for="temp-slider">Temperature: <span id="temp-value">{initial_temperature:.1f}</span></label>
            <input id="temp-slider" data-testid="temp-slider" type="range" min="1" max="16" step="0.1" value="{initial_temperature:.1f}" />
        </div>
    </div>
    <div class="kd-body">
        <img class="preview" src="data:image/png;base64,{image_b64}" alt="sample" />
        <div class="chart" data-testid="soft-target-chart" id="chart-container">
            {bars_html}
        </div>
    </div>
</div>
<script>
(function() {{
    const logits = [{logits_js}];
    const labels = [{labels_js}];
    const slider = document.getElementById('temp-slider');
    const valueLabel = document.getElementById('temp-value');
    const chart = document.getElementById('chart-container');
    function softmax(values, temperature) {{
        const scaled = values.map(v => v / temperature);
        const maxVal = Math.max(...scaled);
        const exps = scaled.map(v => Math.exp(v - maxVal));
        const sum = exps.reduce((acc, value) => acc + value, 0);
        return exps.map(value => value / sum);
    }}
    function render(temperature) {{
        const probs = softmax(logits, temperature);
        const pairs = labels.map((label, index) => [label, probs[index]]).sort((a, b) => b[1] - a[1]);
        chart.innerHTML = pairs.map(([label, prob]) => `
            <div class="bar-row">
                <span class="bar-label">${{label}}</span>
                <div class="bar-track"><div class="bar-fill" style="width:${{(prob * 100).toFixed(2)}}%"></div></div>
                <span class="bar-value">${{prob.toFixed(3)}}</span>
            </div>
        `).join('');
    }}
    slider.addEventListener('input', event => {{
        const temperature = parseFloat(event.target.value);
        valueLabel.textContent = temperature.toFixed(1);
        render(temperature);
    }});
}})();
</script>
"""


def build_dark_knowledge_viewer_html(examples: list[dict]) -> str:
        cards = []
        for example in examples:
                wrongs = example.get("wrong_classes", [])
                wrong_html = "".join(f'<li>{_escape(item["label"])}: {float(item["prob"]):.3f}</li>' for item in wrongs) or "<li>No wrong class above 5%</li>"
                cards.append(
                        f'''
                        <div class="example-card">
                            <img src="data:image/png;base64,{example['image_b64']}" alt="example" />
                            <div>
                                <h4>{_escape(example['label_name'])}</h4>
                                <p>Teacher prediction: <strong>{_escape(example['pred_name'])}</strong> ({float(example['pred_prob']):.3f})</p>
                                <ul>{wrong_html}</ul>
                            </div>
                        </div>
                        '''
                )
        return f"""
<div data-testid="dark-knowledge-viewer" class="kd-card">
    <h3>Dark Knowledge Viewer</h3>
    <p>Examples where the teacher is correct but still assigns meaningful probability to a wrong class.</p>
    <div class="examples-grid">
        {''.join(cards)}
    </div>
</div>
"""


def build_compression_dashboard_html(teacher: dict, student: dict) -> str:
        return f"""
<div data-testid="compression-dashboard" class="kd-card">
    <h3>Compression Dashboard</h3>
    <table class="compare-table">
        <thead>
            <tr><th>Model</th><th>Parameter Count</th><th>Model Size (MB)</th><th>Final Accuracy</th><th>Inference Time (ms)</th></tr>
        </thead>
        <tbody>
            <tr><td>Teacher</td><td>{int(teacher['params'])}</td><td>{float(teacher['model_size_mb']):.2f}</td><td>{float(teacher['accuracy']):.3f}</td><td>{float(teacher['inference_ms']):.2f}</td></tr>
            <tr><td>Student</td><td>{int(student['params'])}</td><td>{float(student['model_size_mb']):.2f}</td><td>{float(student['accuracy']):.3f}</td><td>{float(student['inference_ms']):.2f}</td></tr>
        </tbody>
    </table>
</div>
"""


def build_distillation_curve_html(histories: dict) -> str:
        series_payload = []
        for key, label, color in [
                ("hard_only", "Hard labels only (alpha=1.0)", "#f97316"),
                ("soft_only", "Soft targets only (alpha=0.0)", "#0ea5e9"),
                ("best_alpha", "Best alpha", "#10b981"),
        ]:
                history = histories.get(key, {}).get("history", {})
                points = list(zip(history.get("epoch", []), history.get("test_accuracy", [])))
                series_payload.append({"label": label, "color": color, "points": points})

        serialized_series = json.dumps(series_payload)
        return f"""
<div data-testid="distillation-curve" class="kd-card">
    <h3>Distillation Curve</h3>
    <svg id="distillation-svg" viewBox="0 0 800 360" preserveAspectRatio="none"></svg>
</div>
<script>
(function() {{
    const series = {serialized_series};
    const svg = document.getElementById('distillation-svg');
    const width = 800;
    const height = 360;
    const padding = 48;
    const plotWidth = width - padding * 2;
    const plotHeight = height - padding * 2;
    const allPoints = series.flatMap(s => s.points);
    const maxEpoch = Math.max(...allPoints.map(p => p[0]), 1);
    const maxAcc = Math.max(...allPoints.map(p => p[1]), 1);
    const minAcc = Math.min(...allPoints.map(p => p[1]), 0);
    const yMax = Math.max(maxAcc, 1);
    function xScale(epoch) {{ return padding + ((epoch - 1) / Math.max(maxEpoch - 1, 1)) * plotWidth; }}
    function yScale(acc) {{ return height - padding - ((acc - minAcc) / Math.max(yMax - minAcc, 1e-6)) * plotHeight; }}
    const axis = `
        <line x1="${{padding}}" y1="${{height - padding}}" x2="${{width - padding}}" y2="${{height - padding}}" stroke="#94a3b8" />
        <line x1="${{padding}}" y1="${{padding}}" x2="${{padding}}" y2="${{height - padding}}" stroke="#94a3b8" />
    `;
    const grid = Array.from({{length: 5}}, (_, index) => index / 4).map(f => `
        <line x1="${{padding}}" y1="${{padding + f * plotHeight}}" x2="${{width - padding}}" y2="${{padding + f * plotHeight}}" stroke="#e2e8f0" stroke-dasharray="4 4" />
    `).join('');
    const paths = series.map(entry => {{
        const path = entry.points.map(([epoch, acc], index) => `${{index === 0 ? 'M' : 'L'}} ${{xScale(epoch).toFixed(2)}} ${{yScale(acc).toFixed(2)}}`).join(' ');
        return `
            <path d="${{path}}" fill="none" stroke="${{entry.color}}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" data-series="${{entry.label}}" />
        `;
    }}).join('');
    const legend = series.map((entry, index) => `
        <g transform="translate(${{padding + index * 245}}, 20)">
            <rect width="16" height="16" fill="${{entry.color}}"></rect>
            <text x="24" y="13" font-size="14" fill="#0f172a">${{entry.label}}</text>
        </g>
    `).join('');
    svg.innerHTML = `${{grid}}${{axis}}${{paths}}${{legend}}`;
}})();
</script>
"""

# create a tiny white PNG as placeholder
from PIL import Image
import io

img = Image.new('RGB', (32, 32), color=(255, 255, 255))
buffer = io.BytesIO()
img.save(buffer, format='PNG')
image_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')

# dummy logits (one-hot-like but in logits space)
logits = [0.0] * len(CLASS_NAMES)
logits[0] = 5.0

# load metrics if available
OUT = Path('output')
teacher_metrics = {'params': 450618, 'accuracy': 1.0, 'model_size_mb': 12.34, 'inference_ms': 5.2}
student_metrics = {'params': 88690, 'accuracy': 0.95, 'model_size_mb': 3.2, 'inference_ms': 2.0}
histories = {}
try:
    tpath = OUT / 'teacher' / 'teacher_metrics.json'
    if tpath.exists():
        teacher_metrics.update(load_json(tpath))
    spath = OUT / 'student' / 'best_student_metrics.json'
    if spath.exists():
        student_metrics.update(load_json(spath))
    hpath = OUT / 'student' / 'distillation_histories.json'
    if hpath.exists():
        histories = load_json(hpath)
except Exception:
    pass

soft_html = build_soft_target_inspector_html(image_b64, logits, CLASS_NAMES, 'Soft Target Inspector - preview', initial_temperature=4.0)
dark_html = build_dark_knowledge_viewer_html([
    {
        'image_b64': image_b64,
        'label_name': CLASS_NAMES[0],
        'pred_name': CLASS_NAMES[0],
        'pred_prob': 1.0,
        'wrong_classes': [],
    }
])
comp_html = build_compression_dashboard_html(teacher_metrics, student_metrics)
dist_html = build_distillation_curve_html(histories)

index_html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>KD Preview</title>
<style>body{{font-family:Inter,system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial; padding:20px;}}</style>
</head>
<body>
<h1>Preview: Knowledge Distillation Visualizations</h1>
{soft_html}
<hr>
{dark_html}
<hr>
{comp_html}
<hr>
{dist_html}
</body>
</html>
"""

TEST_DIR = Path('tests')
TEST_DIR.mkdir(exist_ok=True)
(TEST_DIR / 'index.html').write_text(index_html, encoding='utf-8')

# Serve the tests directory on port 8501
server_address = ('', 8501)
handler = SimpleHTTPRequestHandler
httpd = ThreadingHTTPServer(server_address, handler)

def serve():
    print('Serving preview at http://localhost:8501')
    import os
    os.chdir(str(TEST_DIR))
    httpd.serve_forever()

thread = threading.Thread(target=serve, daemon=True)
thread.start()

# Keep main thread alive
try:
    thread.join()
except KeyboardInterrupt:
    httpd.shutdown()
