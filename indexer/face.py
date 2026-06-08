"""InsightFace 人脸检测 + 编码 + 聚类"""

import json
import logging
import uuid
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

import shared.config as cfg

logger = logging.getLogger(__name__)

# 抑制 InsightFace 内部的 scikit-image 废弃警告
warnings.filterwarnings("ignore", message=".*estimate.*deprecated.*", category=FutureWarning)

_face_app = None


def get_face_app():
    """单例获取 InsightFace FaceAnalysis"""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        face_model = cfg.get("face.model", "buffalo_sc")
        device = cfg.resolve_device()
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        _face_app = FaceAnalysis(
            name=face_model,
            providers=providers,
        )
        _face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info("InsightFace %s 就绪 (%s)", face_model, device)
    return _face_app


def detect_faces(image_path: str) -> list[dict]:
    """
    检测一张照片中的所有人脸。

    Returns:
        list of {
            "face_vector": np.ndarray (512,),
            "bbox": str (JSON [x1,y1,x2,y2]),
            "crop": PIL.Image  # 裁剪的人脸
        }
    """
    app = get_face_app()

    img = cv2.imread(image_path)
    if img is None:
        return []

    # InsightFace 用 BGR
    faces = app.get(img)
    if not faces:
        return []

    results = []
    h, w = img.shape[:2]
    min_face_size = int(cfg.get("face.min_face_size", 30))
    for face in faces:
        bbox = face.bbox.astype(int)
        # 裁剪人脸区域（加一点 padding）
        x1, y1, x2, y2 = bbox
        if min(x2 - x1, y2 - y1) < min_face_size:
            continue
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop_rgb = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        results.append({
            "face_vector": face.normed_embedding.astype(np.float32),
            "bbox": json.dumps([int(x1), int(y1), int(x2), int(y2)]),
            "crop": crop_pil,
        })

    return results


def cluster_faces(face_records: list[dict], threshold: Optional[float] = None) -> dict:
    """
    对人脸向量做层次聚类，分配 cluster_id。

    Args:
        face_records: list of {"id": ..., "face_vector": np.ndarray(512,)}
        threshold: 余弦相似度阈值，大于此值归为同一人

    Returns:
        dict mapping face_id -> cluster_id
    """
    if threshold is None:
        threshold = float(cfg.get("face.cluster_threshold", 0.45))

    if not face_records:
        return {}

    n = len(face_records)
    vectors = np.array([r["face_vector"] for r in face_records], dtype=np.float32)

    # 余弦相似度矩阵
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = vectors / norms
    sim_matrix = normed @ normed.T

    # 简单的 union-find 聚类
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                union(i, j)

    # 分配 cluster_id
    clusters = {}
    cluster_map = {}
    for i in range(n):
        root = find(i)
        if root not in cluster_map:
            cluster_map[root] = f"c_{uuid.uuid4().hex[:8]}"
        clusters[face_records[i]["id"]] = cluster_map[root]

    logger.info(f"聚类完成: {n} 张脸 → {len(set(clusters.values()))} 个人")
    return clusters


def save_face_thumbnail(crop: Image.Image, face_id: str, face_dir: Path) -> str:
    """保存人脸缩略图，返回路径"""
    face_dir.mkdir(parents=True, exist_ok=True)
    dest = face_dir / f"{face_id}.webp"
    crop = crop.convert("RGB")
    crop.thumbnail((200, 200))
    crop.save(str(dest), "WEBP", quality=85)
    return str(dest)
