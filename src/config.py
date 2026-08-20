from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dir(cfg, key, sub=None):
    p = PROJECT_ROOT / cfg["data"][key]
    if sub:
        p = p / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
