from pathlib import Path

import pandas as pd


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def code_to_symbol(code: str) -> str:
    """600519 -> sh600519, 000001 -> sz000001"""
    code = str(code).zfill(6)
    if code.startswith(("6", "9", "5")):
        return "sh" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sz" + code


def save_panel(df: pd.DataFrame, path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path)


def load_panel(path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))
