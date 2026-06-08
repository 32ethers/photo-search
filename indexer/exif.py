"""EXIF 元数据提取"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

logger = logging.getLogger(__name__)


def _gps_to_degrees(value) -> float:
    d, m, s = value
    return float(d) + float(m) / 60.0 + float(s) / 3600.0


def _parse_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).isoformat()
        except ValueError:
            continue
    return None


def extract(image_path: str) -> dict:
    """
    提取 EXIF → 结构化元数据

    Returns:
        {
            "date_original": str | None,
            "gps_lat": float | None,
            "gps_lon": float | None,
            "camera_make": str | None,
            "camera_model": str | None,
            "width": int,
            "height": int,
        }
    """
    result = {
        "date_original": None,
        "gps_lat": None,
        "gps_lon": None,
        "camera_make": None,
        "camera_model": None,
        "width": 0,
        "height": 0,
    }

    try:
        img = Image.open(image_path)
        result["width"], result["height"] = img.size
        raw = img._getexif()
        if raw is None:
            return result
    except Exception:
        return result

    exif = {}
    for tag_id, val in raw.items():
        exif[TAGS.get(tag_id, tag_id)] = val

    # 拍摄时间
    date_str = exif.get("DateTimeOriginal") or exif.get("DateTime") or exif.get("DateTimeDigitized")
    result["date_original"] = _parse_date(date_str)

    # GPS
    gps_raw = exif.get("GPSInfo")
    if gps_raw:
        gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
        lat = gps.get("GPSLatitude")
        lon = gps.get("GPSLongitude")
        if lat and lon:
            result["gps_lat"] = _gps_to_degrees(lat)
            if gps.get("GPSLatitudeRef") == "S":
                result["gps_lat"] = -result["gps_lat"]
            result["gps_lon"] = _gps_to_degrees(lon)
            if gps.get("GPSLongitudeRef") == "W":
                result["gps_lon"] = -result["gps_lon"]

    # 相机
    if exif.get("Make"):
        result["camera_make"] = exif["Make"].strip()
    if exif.get("Model"):
        result["camera_model"] = exif["Model"].strip()

    return result
