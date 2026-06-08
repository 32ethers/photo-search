"""配置加载"""

from pathlib import Path
from typing import Any

import torch
import yaml

_cfg: dict = {}


def load(path: str = "config.yaml") -> dict:
    global _cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(p, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    # 确保数据目录存在
    data_dir = Path(get("data_dir", "./data"))
    (data_dir / "thumbnails").mkdir(parents=True, exist_ok=True)
    return _cfg


def get(key: str, default: Any = None) -> Any:
    """点号路径取值: get("models.siglip2")"""
    cur = _cfg
    for k in key.split("."):
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
        if cur is None:
            return default
    return cur


def resolve_device() -> str:
    dev = get("device", "auto")
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return dev
