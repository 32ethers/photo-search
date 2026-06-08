"""LanceDB 存储 - 向量 + 元数据一体"""

import logging
from pathlib import Path
from typing import Optional

import lance
import lancedb
import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)

PHOTOS_TABLE = "photos"
FACES_TABLE = "faces"


def _photos_schema(vector_dim: int = 1152):
    """动态生成 Photos 表 schema，适配不同维度的模型"""
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("file_hash", pa.string()),
        pa.field("thumbnail_path", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        pa.field("date_original", pa.string()),
        pa.field("gps_lat", pa.float32()),
        pa.field("gps_lon", pa.float32()),
        pa.field("gps_city", pa.string()),
        pa.field("gps_country", pa.string()),
        pa.field("camera_make", pa.string()),
        pa.field("camera_model", pa.string()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("indexed_at", pa.string()),
        # 视频扩展字段（照片也兼容）
        pa.field("media_type", pa.string()),      # "photo" / "video"，空=照片
        pa.field("video_duration", pa.float32()),  # 视频秒数，照片=0
    ])

# Faces 表 schema
FACES_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("photo_id", pa.string()),
    pa.field("face_vector", pa.list_(pa.float32(), 512)),
    pa.field("bbox", pa.string()),          # JSON [x1,y1,x2,y2]
    pa.field("thumbnail_path", pa.string()),
    pa.field("cluster_id", pa.string()),
    pa.field("person_name", pa.string()),    # null = 未命名
    pa.field("hidden", pa.bool_()),          # true = 隐藏（路人甲）
])

class Store:
    """照片向量 + 元数据存储"""

    def __init__(self, data_dir: str, vector_dim: int = 1152):
        self.db = lancedb.connect(data_dir)
        self.vector_dim = vector_dim
        self.table: Optional[lancedb.db.LanceTable] = None
        self.faces_table: Optional[lancedb.db.LanceTable] = None
        self._open_or_create()

    def _open_or_create(self):
        # Photos 表
        try:
            self.table = self.db.open_table(PHOTOS_TABLE)
            # 自动检测已有表的向量维度
            sample = self.table.to_pandas(limit=1)
            if not sample.empty and "vector" in sample.columns:
                v = sample.iloc[0]["vector"]
                if hasattr(v, '__len__'):
                    self.vector_dim = len(v)
            logger.info(f"已打开表 {PHOTOS_TABLE}: {self.table.count_rows()} 条, vector_dim={self.vector_dim}")
            # 迁移：检查是否有视频字段，没有则添加
            self._migrate_photos_table()
        except Exception:
            self.table = self.db.create_table(PHOTOS_TABLE, schema=_photos_schema(self.vector_dim))
            logger.info(f"已创建表 {PHOTOS_TABLE} (vector_dim={self.vector_dim})")

        # Faces 表
        try:
            self.faces_table = self.db.open_table(FACES_TABLE)
            logger.info(f"已打开表 {FACES_TABLE}: {self.faces_table.count_rows()} 条记录")
        except Exception:
            self.faces_table = self.db.create_table(FACES_TABLE, schema=FACES_SCHEMA)
            logger.info(f"已创建表 {FACES_TABLE}")

    def _migrate_photos_table(self):
        """给旧表添加视频字段，或处理向量维度变化"""
        try:
            df = self.table.to_pandas(limit=1)
            if df.empty:
                return
            cols = set(df.columns)

            # 检查向量维度是否匹配
            dim_mismatch = False
            if "vector" in cols:
                v = df.iloc[0]["vector"]
                if hasattr(v, '__len__') and len(v) != self.vector_dim:
                    dim_mismatch = True
                    logger.warning(
                        f"向量维度不匹配: 表={len(v)}, 模型={self.vector_dim}, 重建表..."
                    )

            need_migration = (
                dim_mismatch
                or "media_type" not in cols
                or "video_duration" not in cols
            )
            if not need_migration:
                return

            if dim_mismatch:
                # 维度变了，直接重建空表（旧数据无法复用）
                self.db.drop_table(PHOTOS_TABLE)
                self.table = self.db.create_table(PHOTOS_TABLE, schema=_photos_schema(self.vector_dim))
                logger.info(f"表已重建 (vector_dim={self.vector_dim})")
                return

            # 视频字段迁移
            logger.info("迁移 photos 表：添加 video 字段...")
            full_df = self.table.to_pandas()
            if "media_type" not in full_df.columns:
                full_df["media_type"] = ""
            if "video_duration" not in full_df.columns:
                full_df["video_duration"] = 0.0

            records = full_df.to_dict("records")
            for r in records:
                if isinstance(r.get("vector"), np.ndarray):
                    r["vector"] = r["vector"].tolist()
                if not r.get("media_type"):
                    r["media_type"] = ""
                if not r.get("video_duration"):
                    r["video_duration"] = 0.0

            self.db.drop_table(PHOTOS_TABLE)
            self.table = self.db.create_table(PHOTOS_TABLE, data=records)
            logger.info(f"迁移完成: {len(records)} 条记录已添加 video 字段")
        except Exception as e:
            logger.warning(f"photos 表迁移失败: {e}")

    def add(self, record: dict):
        """插入一条照片记录 (含 vector 字段)"""
        # 确保 vector 是 list[float]
        if isinstance(record.get("vector"), np.ndarray):
            record["vector"] = record["vector"].tolist()
        self.table.add([record])

    def search(
        self,
        query_vector: np.ndarray,
        where: Optional[str] = None,
        top_k: int = 30,
    ) -> list[dict]:
        """
        向量搜索 + 元数据过滤

        Args:
            query_vector: SigLIP2 文本向量 (1152,)
            where: LanceDB SQL 过滤条件, 如 "date_original >= '2025-12-01'"
            top_k: 返回条数
        """
        query = self.table.search(query_vector.tolist()).limit(top_k)
        if where:
            query = query.where(where)
        results = query.to_pandas()
        if results.empty:
            return []
        records = []
        for _, row in results.iterrows():
            d = row.to_dict()
            d.pop("_distance", None)
            records.append(d)

        # 从原始向量重新计算余弦相似度
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        for r in records:
            v = np.array(r["vector"], dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if q_norm > 0 and v_norm > 0:
                r["similarity"] = float(np.dot(q, v) / (q_norm * v_norm))
            else:
                r["similarity"] = 0.0
        return records

    def exists(self, file_path: str) -> bool:
        """检查文件是否已索引（单次调用用，慢）"""
        try:
            df = self.table.to_pandas(columns=["file_path"])
            return file_path in df["file_path"].values
        except Exception:
            return False

    def get_indexed_paths(self) -> set[str]:
        """一次性获取所有已索引文件路径（批量检查用，快）"""
        try:
            df = self.table.to_pandas(columns=["file_path"])
            return set(df["file_path"].tolist())
        except Exception:
            return set()

    def filter(self, where: Optional[str] = None, limit: int = 30) -> list[dict]:
        """纯条件过滤（不做向量搜索），按时间倒序"""
        try:
            results = self.table.to_pandas(filter=where)
        except Exception as e:
            logger.warning(f"filter 失败: {e}")
            results = self.table.to_pandas()

        if results.empty:
            return []
        # 按时间倒序排序
        if "date_original" in results.columns:
            results = results.sort_values("date_original", ascending=False)
        results = results.head(limit)
        records = []
        for _, row in results.iterrows():
            d = row.to_dict()
            d["similarity"] = 1.0
            records.append(d)
        return records

    def count(self) -> int:
        return self.table.count_rows()

    def list_all(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """列出照片（管理用）"""
        df = self.table.to_pandas().iloc[offset:offset + limit]
        return df.to_dict("records")

    # ---- Faces 相关 ----

    def add_faces(self, face_records: list[dict]):
        """批量插入人脸记录"""
        for r in face_records:
            if isinstance(r.get("face_vector"), np.ndarray):
                r["face_vector"] = r["face_vector"].tolist()
        self.faces_table.add(face_records)

    def get_faces_for_photo(self, photo_id: str) -> list[dict]:
        """获取一张照片的所有人脸"""
        try:
            df = self.faces_table.to_pandas(filter=f"photo_id = '{photo_id}'")
        except Exception:
            return []
        if df.empty:
            return []
        return df.to_dict("records")

    def get_all_faces(self) -> list[dict]:
        """获取所有人脸记录"""
        try:
            df = self.faces_table.to_pandas()
        except Exception:
            return []
        if df.empty:
            return []
        return df.to_dict("records")

    def get_clusters(self) -> list[dict]:
        """获取所有聚类（含代表脸、照片数、名字、隐藏状态）"""
        df = self.faces_table.to_pandas()
        if df.empty:
            return []

        # 兼容旧表无 hidden 列
        if "hidden" not in df.columns:
            df["hidden"] = False

        clusters = {}
        for _, row in df.iterrows():
            cid = row.get("cluster_id", "")
            if not cid:
                continue
            if cid not in clusters:
                clusters[cid] = {
                    "cluster_id": cid,
                    "person_name": row.get("person_name") or None,
                    "photo_ids": set(),
                    "face_count": 0,
                    "representative_thumb": row.get("thumbnail_path", ""),
                    "representative_id": row.get("id", ""),
                    "hidden": False,
                }
            clusters[cid]["photo_ids"].add(row["photo_id"])
            clusters[cid]["face_count"] += 1
            # 只要有一条 hidden 就算隐藏
            if row.get("hidden"):
                clusters[cid]["hidden"] = True

        result = []
        for c in clusters.values():
            c["photo_count"] = len(c["photo_ids"])
            del c["photo_ids"]
            result.append(c)
        # 按照片数降序
        result.sort(key=lambda x: x["photo_count"], reverse=True)
        return result

    def get_photos_by_cluster(
        self, cluster_id: str,
        date_from: str = None, date_to: str = None,
        location: str = None, device: str = None,
    ) -> list[dict]:
        """获取某人物的所有照片（含人脸信息），支持筛选"""
        try:
            faces_df = self.faces_table.to_pandas(filter=f"cluster_id = '{cluster_id}'")
        except Exception:
            return []
        if faces_df.empty:
            return []

        photo_ids = set(faces_df["photo_id"].unique().tolist())
        # 从 photos 表取照片详情
        photos_df = self.table.to_pandas()
        photos_df = photos_df[photos_df["id"].isin(photo_ids)]

        # 筛选
        if date_from:
            photos_df = photos_df[photos_df["date_original"].fillna("") >= date_from]
        if date_to:
            photos_df = photos_df[photos_df["date_original"].fillna("") <= date_to]
        if location:
            photos_df = photos_df[photos_df["gps_city"].fillna("").str.contains(location, na=False)]
        if device:
            mask = (
                photos_df["camera_make"].fillna("").str.contains(device, case=False, na=False) |
                photos_df["camera_model"].fillna("").str.contains(device, case=False, na=False)
            )
            photos_df = photos_df[mask]

        # 为每张照片附加人脸 bbox 信息
        face_bboxes = {}
        for _, fr in faces_df.iterrows():
            pid = fr["photo_id"]
            if pid not in face_bboxes:
                face_bboxes[pid] = []
            face_bboxes[pid].append({
                "face_id": fr["id"],
                "bbox": fr.get("bbox", ""),
                "person_name": fr.get("person_name") or None,
                "cluster_id": fr.get("cluster_id", ""),
            })

        records = []
        for _, row in photos_df.iterrows():
            d = row.to_dict()
            d.pop("vector", None)
            d["similarity"] = 1.0
            d["faces"] = face_bboxes.get(d["id"], [])
            records.append(d)

        # 按时间倒序
        records.sort(key=lambda x: x.get("date_original", ""), reverse=True)
        return records

    def _overwrite_faces(self, df):
        """用 DataFrame 覆写 faces 表"""
        records = df.to_dict("records")
        for r in records:
            if isinstance(r.get("face_vector"), np.ndarray):
                r["face_vector"] = r["face_vector"].tolist()
            # 确保 hidden 是 bool
            if "hidden" not in r or r["hidden"] is None:
                r["hidden"] = False
            elif not isinstance(r["hidden"], (bool, np.bool_)):
                r["hidden"] = bool(r["hidden"])
        self.db.drop_table(FACES_TABLE)
        self.faces_table = self.db.create_table(FACES_TABLE, data=records)

    def update_cluster_name(self, cluster_id: str, name: str):
        """更新聚类中所有人脸的名字"""
        df = self.faces_table.to_pandas()
        if df.empty:
            return
        mask = df["cluster_id"] == cluster_id
        df.loc[mask, "person_name"] = name
        self._overwrite_faces(df)

    def face_exists_for_photo(self, photo_id: str) -> bool:
        """检查照片是否已做过人脸检测"""
        try:
            df = self.faces_table.to_pandas(filter=f"photo_id = '{photo_id}'")
            return len(df) > 0
        except Exception:
            return False

    def count_faces(self) -> int:
        return self.faces_table.count_rows()

    def merge_clusters(self, source_ids: list[str], target_id: str, name: str = ""):
        """
        合并多个聚类为一个。
        source_ids 里除 target_id 之外的 cluster 统一指向 target_id。
        """
        df = self.faces_table.to_pandas()
        if df.empty:
            return

        for sid in source_ids:
            if sid == target_id:
                continue
            mask = df["cluster_id"] == sid
            df.loc[mask, "cluster_id"] = target_id

        # 更新名字
        if name:
            mask = df["cluster_id"] == target_id
            df.loc[mask, "person_name"] = name

        self._overwrite_faces(df)

    def set_cluster_hidden(self, cluster_id: str, hidden: bool):
        """设置聚类的隐藏状态"""
        df = self.faces_table.to_pandas()
        if df.empty:
            return
        # 列可能没有 hidden（旧数据）
        if "hidden" not in df.columns:
            df["hidden"] = False
        mask = df["cluster_id"] == cluster_id
        df.loc[mask, "hidden"] = hidden
        self._overwrite_faces(df)
