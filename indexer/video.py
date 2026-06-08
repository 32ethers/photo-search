"""视频处理：FFmpeg 采帧 + 选最佳帧"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> dict:
    """获取视频元数据"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {}
    data = json.loads(result.stdout)
    fmt = data.get("format", {})
    info = {
        "duration": float(fmt.get("duration", 0)),
        "width": 0,
        "height": 0,
    }
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            info["width"] = s.get("width", 0)
            info["height"] = s.get("height", 0)
            break
    return info


def extract_frames(
    video_path: str,
    info: Optional[dict] = None,
    max_seconds: float = 30.0,
    interval: float = 2.0,
) -> list[dict]:
    """
    从视频前 max_seconds 秒中每隔 interval 秒提取一帧。

    Returns:
        [{"time": 0.0, "image": PIL.Image}, ...]
    """
    info = info or get_video_info(video_path)
    duration = info.get("duration", 0)
    if duration <= 0:
        return []

    end_time = min(duration, max_seconds)

    # 用 ffmpeg 提取帧到临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 短视频直接取第一帧
        if duration <= interval:
            cmd = [
                "ffmpeg", "-v", "quiet",
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                os.path.join(tmpdir, "frame_0001.jpg"),
            ]
        else:
            cmd = [
                "ffmpeg", "-v", "quiet",
                "-ss", "0",
                "-i", video_path,
                "-t", str(end_time),
                "-vf", f"fps=1/{interval}",
                "-q:v", "2",
                os.path.join(tmpdir, "frame_%04d.jpg"),
            ]
        subprocess.run(cmd, capture_output=True, timeout=120)

        frames = []
        for i, f in enumerate(sorted(Path(tmpdir).glob("frame_*.jpg"))):
            t = i * interval
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            frames.append({"time": t, "image": img})

    return frames


def select_best_frame(
    frames: list[dict],
    video_path: str = "",
    detect_max_side: int = 960,
) -> dict | None:
    """
    从一组帧中选最有意义的那一帧。

    策略：选人脸最多（且最大）的帧。没有人脸则取中间帧。
    """
    if not frames:
        return None

    # 尝试用 InsightFace 检测人脸
    try:
        from indexer.face import get_face_app
        import cv2

        app = get_face_app()
        best_frame = None
        best_score = -1  # 人脸数量 × 最大人脸面积

        for frame in frames:
            img = frame["image"]
            width, height = img.size
            scale = 1.0
            detect_img = img
            if max(width, height) > detect_max_side:
                scale = detect_max_side / float(max(width, height))
                detect_img = img.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.Resampling.BILINEAR,
                )
            # PIL → BGR for InsightFace
            arr = np.array(detect_img)
            bgr = arr[:, :, ::-1].copy()
            faces = app.get(bgr)
            if faces:
                # 得分 = 人脸数量 × 最大人脸面积
                areas = []
                for f in faces:
                    x1, y1, x2, y2 = f.bbox
                    face_area = (x2 - x1) * (y2 - y1)
                    if scale != 1.0:
                        face_area = face_area / (scale * scale)
                    areas.append(face_area)
                score = len(faces) * max(areas) if areas else 0
                if score > best_score:
                    best_score = score
                    best_frame = frame
                    best_frame["face_count"] = len(faces)

        if best_frame:
            logger.info(
                f"选中帧 t={best_frame['time']:.1f}s, "
                f"face_count={best_frame.get('face_count', 0)}"
            )
            return best_frame

    except Exception as e:
        logger.warning(f"人脸选帧失败，回退到中间帧: {e}")

    # 没有人脸，取中间帧
    mid = len(frames) // 2
    logger.info(f"无人脸，取中间帧 t={frames[mid]['time']:.1f}s")
    return frames[mid]


def process_video(
    video_path: str,
    max_seconds: float = 30.0,
    interval: float = 2.0,
) -> dict | None:
    """
    处理一个视频文件：采帧 → 选最佳帧 → 返回。

    Returns:
        {"image": PIL.Image, "duration": float, "info": dict}
    """
    info = get_video_info(video_path)
    if info.get("duration", 0) <= 0:
        logger.warning(f"视频无法读取: {video_path}")
        return None

    logger.info(
        f"处理视频: {Path(video_path).name} "
        f"({info['duration']:.1f}s, {info['width']}x{info['height']})"
    )

    frames = extract_frames(video_path, info=info, max_seconds=max_seconds, interval=interval)
    if not frames:
        logger.warning(f"无法提取帧: {video_path}")
        return None

    best = select_best_frame(frames, video_path)
    if not best:
        return None

    return {
        "image": best["image"],
        "duration": info["duration"],
        "info": info,
    }
