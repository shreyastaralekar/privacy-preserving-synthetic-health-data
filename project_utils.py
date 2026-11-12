"""Shared local paths and console display helpers for the BRFSS project."""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_DIRS = (
    "data", "plots", "checkpoints", "models", "metrics",
    "artifacts", "artifacts/synthetic_data", "final_deliverables", "logs"
)

def configure_project(script_file: str | Path) -> Path:
    """Return the project root; defaults to the parent of the scripts folder.

    Override with DP_PROJECT_ROOT if data should live somewhere else.
    """
    override = os.environ.get("DP_PROJECT_ROOT", "").strip()
    if override:
        root = Path(override).expanduser().resolve()
    else:
        script_path = Path(script_file).resolve()
        root = script_path.parent.parent if script_path.parent.name == "scripts" else script_path.parent
    root.mkdir(parents=True, exist_ok=True)
    for rel in PROJECT_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root

def display(*objects, max_rows: int = 30, max_cols: int = 24) -> None:
    """Notebook-like display fallback for ordinary Python IDE consoles."""
    for obj in objects:
        if hasattr(obj, "to_string"):
            try:
                print(obj.to_string(max_rows=max_rows, max_cols=max_cols))
            except TypeError:
                print(obj.to_string())
        else:
            print(obj)

def print_environment() -> None:
    import sys
    print("Python:", sys.version.split()[0])
    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("PyTorch check unavailable:", exc)
