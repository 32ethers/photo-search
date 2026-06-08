"""查询解析器 - 规则引擎，不依赖 LLM"""

import logging
import re
from datetime import datetime, date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_CITIES = {
    "北京", "上海", "广州", "深圳", "成都", "杭州", "重庆", "武汉", "西安",
    "天津", "南京", "苏州", "长沙", "郑州", "青岛", "大连", "厦门", "宁波",
    "济南", "福州", "合肥", "昆明", "贵阳", "南昌", "太原", "石家庄", "哈尔滨",
    "长春", "沈阳", "承德", "秦皇岛", "张家口", "唐山", "保定", "邯郸",
    "通州", "朝阳", "海淀", "丰台", "怀柔", "顺义", "昌平", "大兴",
    "香港", "澳门", "台北", "桂林", "三亚", "海口", "洛阳", "开封", "敦煌",
}

_DEVICES = {
    "iphone": "iPhone", "ipad": "iPad",
    "samsung": "Samsung", "galaxy": "Samsung",
    "华为": "华为", "huawei": "华为",
    "小米": "小米", "xiaomi": "小米",
    "oppo": "OPPO", "vivo": "VIVO",
    "canon": "Canon", "nikon": "Nikon", "sony": "Sony",
    "大疆": "大疆", "dji": "大疆",
}

_GENERIC = {"照片", "图片", "相片", "所有", "全部", "看看", "帮我找", "找一下"}


def _parse_time(query, now):
    year = now.year
    m = re.search(r"(\d{4})年(\d{1,2})月", query)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        first = date(y, mo, 1)
        last = (date(y, mo + 1, 1) - timedelta(days=1)) if mo < 12 else date(y, 12, 31)
        return first.isoformat(), last.isoformat()
    base = None
    if "去年" in query:
        base = year - 1
    elif "前年" in query:
        base = year - 2
    elif "今年" in query:
        base = year
    _CN_MONTHS = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"十一":11,"十二":12}
    if base:
        mo = None
        # 先匹配中文数字月（十一/十二 在前面）
        for cn, num in sorted(_CN_MONTHS.items(), key=lambda x: -len(x[0])):
            if cn + "月" in query:
                mo = num
                break
        if mo is None:
            m = re.search(r"(\d{1,2})月", query)
            if m:
                mo = int(m.group(1))
        if mo:
            first = date(base, mo, 1)
            last = (date(base, mo + 1, 1) - timedelta(days=1)) if mo < 12 else date(base, 12, 31)
            return first.isoformat(), last.isoformat()
        return f"{base}-01-01", f"{base}-12-31"
    if "上个月" in query:
        first_this = date(now.year, now.month, 1)
        last_prev = first_this - timedelta(days=1)
        return date(last_prev.year, last_prev.month, 1).isoformat(), last_prev.isoformat()

    # 兜底：单独的 "X月" 或 "X月"（无年份前缀）→ 当年
    _CN_MONTHS = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"十一":11,"十二":12}
    for cn, num in sorted(_CN_MONTHS.items(), key=lambda x: -len(x[0])):
        if cn + "月" in query:
            first = date(year, num, 1)
            last = (date(year, num + 1, 1) - timedelta(days=1)) if num < 12 else date(year, 12, 31)
            return first.isoformat(), last.isoformat()
    m = re.search(r"(\d{1,2})月", query)
    if m:
        mo = int(m.group(1))
        if 1 <= mo <= 12:
            first = date(year, mo, 1)
            last = (date(year, mo + 1, 1) - timedelta(days=1)) if mo < 12 else date(year, 12, 31)
            return first.isoformat(), last.isoformat()

    return None, None


def _extract_location(query):
    # 1. 先匹配 "XX区/市/省/县" 模式（2-6个汉字 + 行政区划后缀）
    m = re.search(r"([\u4e00-\u9fff]{2,6})(?:区|市|省|县|镇|乡)", query)
    if m:
        return m.group(0)
    # 2. 从城市表匹配
    for city in sorted(_CITIES, key=len, reverse=True):
        if city in query:
            return city
    return None


def _extract_device(query):
    lower = query.lower()
    for key, brand in _DEVICES.items():
        if key in lower:
            return brand
    return None


def _extract_visual(query, location, device):
    text = query
    # 去时间
    for p in [r"\d{4}年\d{1,2}月(?:\d{1,2}日)?", r"去年|前年|今年|上个月", r"\d{1,2}月", r"[一二三四五六七八九十]+月"]:
        text = re.sub(p, "", text)
    # 去地点（同时去掉后面的"区/市/省/县"）
    if location:
        text = re.sub(re.escape(location) + r"[区市省县]?", "", text)
    # 去设备
    if device:
        text = re.sub(re.escape(device), "", text, flags=re.IGNORECASE)
    # 去通用词
    for g in _GENERIC:
        text = text.replace(g, "")
    # 去虚词/助词/常见动词
    text = re.sub(r"[在用的拍于了摄到从把被拍摄拍下看看找]", "", text)
    text = re.sub(r"照片|图片|相片", "", text)
    text = re.sub(r"\s+", "", text).strip()
    return text if len(text) >= 1 else ""


class _Parser:
    def parse(self, query):
        now = datetime.now()
        df, dt = _parse_time(query, now)
        loc = _extract_location(query)
        dev = _extract_device(query)
        tq = _extract_visual(query, loc, dev)
        result = {"text_query": tq, "date_from": df, "date_to": dt, "location": loc, "device": dev}
        logger.info(f"解析: {query!r} -> {result}")
        return result


_parser = None

def get_parser():
    global _parser
    if _parser is None:
        _parser = _Parser()
    return _parser
