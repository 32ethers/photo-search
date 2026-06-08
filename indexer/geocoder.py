"""GPS 反向地理编码"""

import json
import logging
import socket
from pathlib import Path
from typing import Optional

import requests
from geopy.geocoders import Nominatim

import shared.config as cfg

logger = logging.getLogger(__name__)

_cache: dict = {}
_cache_path: Optional[Path] = None
_geolocator = None
_unavailable = False


def _init():
    global _geolocator, _cache_path, _unavailable
    if _geolocator is not None:
        return

    _cache_path = Path(cfg.get("data_dir", "./data")) / "geocode_cache.json"
    if _cache_path.exists():
        try:
            _cache.update(json.loads(_cache_path.read_text()))
        except Exception:
            pass

    proxy = cfg.get("proxy", "")
    proxies = {"http": proxy, "https": proxy} if proxy else {}

    # 探测网络是否可达
    try:
        requests.get("https://nominatim.openstreetmap.org", timeout=5, proxies=proxies)
    except Exception:
        _unavailable = True
        logger.warning("地理编码不可用（nominatim.openstreetmap.org 无法连接），跳过 GPS 编码")
        return

    _geolocator = Nominatim(
        user_agent="photo-search",
        timeout=5,
        proxies=proxies,
    )


def reverse(lat: float, lon: float) -> tuple[Optional[str], Optional[str]]:
    """经纬度 → (城市, 国家), 带缓存"""
    global _unavailable
    _init()

    key = f"{lat:.2f},{lon:.2f}"
    if key in _cache:
        return _cache[key].get("city"), _cache[key].get("country")

    if _unavailable or _geolocator is None:
        return None, None

    try:
        loc = _geolocator.reverse((lat, lon), language="zh", timeout=5)
        if loc and loc.raw.get("address"):
            addr = loc.raw["address"]
            city = addr.get("city") or addr.get("town") or addr.get("county") or addr.get("state")
            country = addr.get("country")
            _cache[key] = {"city": city, "country": country}
            _save()
            return city, country
    except Exception:
        if not _unavailable:
            _unavailable = True
            logger.warning("地理编码请求失败，后续跳过 GPS 编码")
    return None, None


def _save():
    if _cache_path:
        _cache_path.write_text(json.dumps(_cache, ensure_ascii=False, indent=2))
