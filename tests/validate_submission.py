import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sub = ROOT / 'submission.json'
if not sub.exists():
    print('submission.json not found')
    sys.exit(2)

data = json.loads(sub.read_text())
errors = False
for key, rel in data.items():
    path = ROOT / rel
    if not path.exists():
        print(f'MISSING: {key} -> {path}')
        errors = True
    else:
        size = path.stat().st_size
        if size == 0:
            print(f'EMPTY: {key} -> {path} (0 bytes)')
            errors = True
        else:
            print(f'OK: {key} -> {path} ({size} bytes)')

if errors:
    print('Validation FAILED')
    sys.exit(1)
else:
    print('Validation PASSED')
    sys.exit(0)
