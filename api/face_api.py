"""人脸相关 API"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import shared.config as cfg
from shared.store import Store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/faces", tags=["faces"])


def _store() -> Store:
    return Store(cfg.get("data_dir", "./data"))


@router.get("/clusters")
async def list_clusters():
    """获取所有人物聚类"""
    store = _store()
    clusters = store.get_clusters()
    # 清理 numpy 类型
    for c in clusters:
        for k, v in c.items():
            if hasattr(v, "item"):
                c[k] = v.item()
    return {"clusters": clusters}


class RenameRequest(BaseModel):
    name: str


@router.put("/clusters/{cluster_id}")
async def rename_cluster(cluster_id: str, req: RenameRequest):
    """给人物聚类命名"""
    store = _store()
    try:
        store.update_cluster_name(cluster_id, req.name)
        return {"ok": True, "cluster_id": cluster_id, "name": req.name}
    except Exception as e:
        logger.exception("重命名失败")
        raise HTTPException(500, str(e))


@router.get("/clusters/{cluster_id}/photos")
async def cluster_photos(
    cluster_id: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    location: Optional[str] = None,
    device: Optional[str] = None,
):
    """获取某人物的所有照片，支持按时间/地点/设备筛选"""
    store = _store()
    import numpy as np
    photos = store.get_photos_by_cluster(
        cluster_id,
        date_from=date_from, date_to=date_to,
        location=location, device=device,
    )
    # 清理
    for p in photos:
        for k in list(p.keys()):
            v = p[k]
            if isinstance(v, (float,)):
                import math
                if math.isnan(v):
                    p[k] = None
            elif hasattr(v, "item"):
                p[k] = v.item()
        # 清理 faces 里的 numpy 类型
        for f in p.get("faces", []):
            for fk, fv in f.items():
                if hasattr(fv, "item"):
                    f[fk] = fv.item()
    return {"photos": photos, "total": len(photos)}


@router.get("/photos/{photo_id}")
async def photo_faces(photo_id: str):
    """获取一张照片的所有人脸"""
    store = _store()
    faces = store.get_faces_for_photo(photo_id)
    import numpy as np
    for f in faces:
        f.pop("face_vector", None)
        for k, v in f.items():
            if isinstance(v, np.floating):
                f[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, np.integer):
                f[k] = int(v)
            elif hasattr(v, "item"):
                f[k] = v.item()
    return {"faces": faces}


class MergeRequest(BaseModel):
    cluster_ids: list[str]
    name: str = ""


@router.post("/clusters/merge")
async def merge_clusters(req: MergeRequest):
    """合并多个聚类为同一个人物"""
    if len(req.cluster_ids) < 2:
        raise HTTPException(400, "至少选择2个聚类")
    store = _store()
    try:
        target_id = req.cluster_ids[0]
        store.merge_clusters(req.cluster_ids, target_id, req.name)
        return {"ok": True, "target_cluster_id": target_id, "name": req.name}
    except Exception as e:
        logger.exception("合并失败")
        raise HTTPException(500, str(e))


class HideRequest(BaseModel):
    hidden: bool


@router.put("/clusters/{cluster_id}/hidden")
async def set_hidden(cluster_id: str, req: HideRequest):
    """隐藏/取消隐藏一个聚类"""
    store = _store()
    try:
        store.set_cluster_hidden(cluster_id, req.hidden)
        return {"ok": True, "cluster_id": cluster_id, "hidden": req.hidden}
    except Exception as e:
        logger.exception("隐藏操作失败")
        raise HTTPException(500, str(e))


@router.post("/reindex")
async def reindex_faces():
    """重新跑人脸检测+聚类"""
    try:
        from indexer.face import detect_faces, cluster_faces, save_face_thumbnail, get_face_app
        from pathlib import Path
        import uuid
        import numpy as np

        store = _store()
        face_dir = Path(cfg.get("data_dir", "./data")) / "faces"

        # 先确保模型加载
        get_face_app()

        # 获取所有已索引的照片
        photos_df = store.table.to_pandas()
        if photos_df.empty:
            return {"status": "no photos"}

        all_face_records = []
        face_data = []  # 用于聚类

        for _, row in photos_df.iterrows():
            photo_id = row["id"]
            file_path = row["file_path"]
            media_type = row.get("media_type", "")
            thumb_path = row.get("thumbnail_path", "")

            import os

            # 视频文件 InsightFace 读不了，用缩略图（即最佳帧）做人脸检测
            if media_type == "video":
                if thumb_path and os.path.exists(thumb_path):
                    detect_target = thumb_path
                else:
                    continue
            else:
                if not os.path.exists(file_path):
                    continue
                detect_target = file_path

            faces = detect_faces(detect_target)
            for face_info in faces:
                face_id = str(uuid.uuid4())
                thumb_path = save_face_thumbnail(face_info["crop"], face_id, face_dir)

                record = {
                    "id": face_id,
                    "photo_id": photo_id,
                    "face_vector": face_info["face_vector"].tolist(),
                    "bbox": face_info["bbox"],
                    "thumbnail_path": thumb_path,
                    "cluster_id": "",  # 稍后填充
                    "person_name": "",
                    "hidden": False,
                }
                all_face_records.append(record)
                face_data.append({
                    "id": face_id,
                    "face_vector": face_info["face_vector"],
                })

        # 聚类
        cluster_map = cluster_faces(face_data)
        for r in all_face_records:
            r["cluster_id"] = cluster_map.get(r["id"], "c_unknown")

        # 清空旧数据并写入新数据
        try:
            store.db.drop_table("faces")
        except Exception:
            pass
        from shared.store import FACES_TABLE, FACES_SCHEMA
        store.faces_table = store.db.create_table(FACES_TABLE, schema=FACES_SCHEMA)
        if all_face_records:
            store.add_faces(all_face_records)

        n_clusters = len(set(r["cluster_id"] for r in all_face_records))
        return {
            "status": "ok",
            "faces_detected": len(all_face_records),
            "clusters": n_clusters,
        }

    except Exception as e:
        logger.exception("人脸重索引失败")
        raise HTTPException(500, str(e))
