"""搜索逻辑"""

import logging
from typing import Optional

import numpy as np

import shared.config as cfg
from shared.store import Store

logger = logging.getLogger(__name__)


def _build_where(date_from, date_to, location, device):
    clauses = []
    if date_from:
        clauses.append(f"date_original >= '{date_from}'")
    if date_to:
        clauses.append(f"date_original <= '{date_to}'")
    if location:
        clauses.append(f"gps_city LIKE '%{location}%'")
    if device:
        tokens = [t for t in device.split() if t]
        if tokens:
            parts = [f"(camera_make LIKE '%{t}%' OR camera_model LIKE '%{t}%')" for t in tokens]
            clauses.append("(" + " AND ".join(parts) + ")")
    if not clauses:
        return None
    return " AND ".join(clauses)

def search(
    text_query: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    location: Optional[str] = None,
    device: Optional[str] = None,
    top_k: int = 30,
    offset: int = 0,
) -> dict:
    """直接搜索：文本 → 向量 + 条件过滤，支持 offset 分页"""
    store = Store(cfg.get("data_dir", "./data"))
    text_query = (text_query or "").strip()
    where = _build_where(date_from, date_to, location, device)
    logger.info("text=%r, where=%s, offset=%s", text_query, where, offset)

    # 多取 1 条，前端分页基于 has_more，而不是猜测“刚好满一页就还有更多”
    fetch_k = top_k + offset + 1

    if text_query:
        from indexer.encoder import get_encoder
        encoder = get_encoder()
        vec = encoder.encode_text(text_query)
        results = store.search(vec, where=where, top_k=fetch_k)
    else:
        results = store.filter(where=where, limit=fetch_k)

    has_more = len(results) > offset + top_k

    # 跳过 offset 条
    results = results[offset:offset + top_k]

    # 去掉 vector 字段，清理 NaN 和 numpy 类型
    for r in results:
        r.pop("vector", None)
        for k in list(r.keys()):
            v = r[k]
            if isinstance(v, (float, np.floating)):
                if np.isnan(v) if isinstance(v, np.floating) else v != v:
                    r[k] = None
                else:
                    r[k] = float(v)
            elif isinstance(v, np.integer):
                r[k] = int(v)

    total = None if has_more else offset + len(results)
    return {
        "results": results,
        "total": total,
        "has_more": has_more,
        "offset": offset,
        "limit": top_k,
    }
