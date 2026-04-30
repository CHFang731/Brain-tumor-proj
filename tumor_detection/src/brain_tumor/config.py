from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return data


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def configure_runtime(root: str | Path = ".") -> None:
    root_path = Path(root)
    mpl_config = root_path / ".cache" / "matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))


def require_dependency(module_name: str, package_name: str | None = None) -> None:
    try:
        __import__(module_name)
    except ImportError as exc:
        package = package_name or module_name
        raise SystemExit(
            f"Missing dependency '{package}'. Install project requirements with: "
            f".venv/bin/python -m pip install -r requirements.txt"
        ) from exc
