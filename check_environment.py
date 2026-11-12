#!/usr/bin/env python
"""Check local Python dependencies, accelerator status and dataset placement."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
print('Project root:', ROOT)
print('Python:', sys.version.split()[0])
failed = []
checks = [
    ('numpy', 'numpy'), ('pandas', 'pandas'), ('matplotlib', 'matplotlib'),
    ('scikit-learn', 'sklearn'), ('scipy', 'scipy'), ('Opacus', 'opacus'),
    ('python-docx', 'docx'), ('python-pptx', 'pptx'),
]
for label, module in checks:
    try:
        imported = __import__(module)
        print(f'[OK] {label}: {getattr(imported, "__version__", "installed")}')
    except Exception as exc:
        print(f'[MISSING] {label}: {exc}')
        failed.append(label)
try:
    import torch
    print('[OK] PyTorch:', torch.__version__)
    print('[INFO] CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('[INFO] GPU:', torch.cuda.get_device_name(0))
except Exception as exc:
    print('[MISSING] PyTorch:', exc)
    failed.append('PyTorch')
xpt = sorted(list((ROOT / 'data').glob('*.XPT')) + list((ROOT / 'data').glob('*.xpt')) + list((ROOT / 'data').glob('*.xport')))
if xpt:
    for p in xpt:
        print('[OK] CDC XPT:', p.name, f'({p.stat().st_size / (1024**2):.1f} MB)')
else:
    print('[ACTION REQUIRED] Put extracted CDC 2025 LLCP2025.XPT in data/')
print('\nCheck complete.')
if failed:
    print('Install missing libraries after activating the project virtual environment.')
    raise SystemExit(1)
