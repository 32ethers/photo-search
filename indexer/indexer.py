"""索引主逻辑 - 扫描目录 + 提取 + 编码 + 存储"""

import hashlib
import logging
import os
import queue
import signal
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

import shared.config as cfg
from shared.store import Store
from indexer import exif, geocoder
from indexer.encoder import get_encoder

logger = logging.getLogger(__name__)

_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n⚠️  收到中断信号，正在优雅退出（等待当前文件处理完毕）...")


signal.signal(signal.SIGINT, _signal_handler)


class Indexer:
    def __init__(self):
        print("📂 初始化索引服务...")

        data_dir = cfg.get("data_dir", "./data")
        print(f"   数据目录: {Path(data_dir).resolve()}")

        print("   加载 SigLIP2 模型...", end=" ", flush=True)
        self.encoder = get_encoder()
        print("就绪")

        # 用模型实际输出维度创建/打开数据库
        vector_dim = self.encoder.encode_text("test").shape[0]
        print(f"   向量维度: {vector_dim}")

        print("   打开数据库...", end=" ", flush=True)
        self.store = Store(data_dir, vector_dim=vector_dim)
        existing = self.store.count()
        print(f"已入库 {existing} 张")

        self.thumb_dir = Path(data_dir) / "thumbnails"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.thumb_size = tuple(cfg.get("thumbnail_size", [300, 300]))
        self.exts = set(cfg.get("supported_extensions", [".jpg", ".jpeg", ".png", ".webp"]))
        self.video_exts = set(cfg.get("video_extensions", [".mp4", ".mov", ".avi", ".3gp", ".mkv"]))
        # 线程数：CPU 数量 - 1，最少 2
        self.workers = max(2, os.cpu_count() - 1)
        print(f"   线程数: {self.workers}")

    def _hash(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _thumbnail(self, image_path: str, photo_id: str) -> str:
        dest = self.thumb_dir / f"{photo_id}.webp"
        if dest.exists():
            return str(dest)
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail(self.thumb_size)
                img.save(str(dest), "WEBP", quality=85)
        except Exception as e:
            logger.warning(f"缩略图失败 {image_path}: {e}")
            return ""
        return str(dest)

    def index_file(self, image_path: str) -> bool:
        """索引单个文件，返回是否成功"""
        path = Path(image_path)
        abs_path = str(path.resolve())

        if self.store.exists(abs_path):
            return False

        try:
            meta = exif.extract(abs_path)
            photo_id = str(uuid.uuid4())
            thumb = self._thumbnail(abs_path, photo_id)
            vector = self.encoder.encode_image(abs_path)

            city, country = None, None
            if meta["gps_lat"] is not None and meta["gps_lon"] is not None:
                city, country = geocoder.reverse(meta["gps_lat"], meta["gps_lon"])

            record = {
                "id": photo_id,
                "file_path": abs_path,
                "file_hash": self._hash(abs_path),
                "thumbnail_path": thumb,
                "vector": vector.tolist(),
                "date_original": meta["date_original"] or "",
                "gps_lat": meta["gps_lat"],
                "gps_lon": meta["gps_lon"],
                "gps_city": city or "",
                "gps_country": country or "",
                "camera_make": meta["camera_make"] or "",
                "camera_model": meta["camera_model"] or "",
                "width": meta["width"],
                "height": meta["height"],
                "indexed_at": datetime.now().isoformat(),
            }
            self.store.add(record)
            return True

        except Exception as e:
            logger.error(f"索引失败 {image_path}: {e}")
            return False

    def index_video(self, video_path: str) -> bool:
        """索引单个视频文件，返回是否成功"""
        path = Path(video_path)
        abs_path = str(path.resolve())

        if self.store.exists(abs_path):
            return False

        try:
            from indexer.video import process_video

            result = process_video(abs_path)
            if not result:
                return False

            photo_id = str(uuid.uuid4())
            best_frame = result["image"]
            duration = result["duration"]
            info = result["info"]

            # 缩略图来自最佳帧
            thumb = self._thumbnail_from_pil(best_frame, photo_id)
            vector = self.encoder.encode_image(best_frame)

            # 视频的 EXIF 信息通常从文件名或目录推断
            meta = exif.extract(abs_path) if Path(abs_path).suffix.lower() in self.exts else {}

            record = {
                "id": photo_id,
                "file_path": abs_path,
                "file_hash": self._hash(abs_path),
                "thumbnail_path": thumb,
                "vector": vector.tolist(),
                "date_original": meta.get("date_original") or "",
                "gps_lat": meta.get("gps_lat"),
                "gps_lon": meta.get("gps_lon"),
                "gps_city": meta.get("gps_city") or "",
                "gps_country": meta.get("gps_country") or "",
                "camera_make": meta.get("camera_make") or "",
                "camera_model": meta.get("camera_model") or "",
                "width": info.get("width", 0),
                "height": info.get("height", 0),
                "indexed_at": datetime.now().isoformat(),
                "media_type": "video",
                "video_duration": duration,
            }
            self.store.add(record)
            return True

        except Exception as e:
            logger.error(f"视频索引失败 {video_path}: {e}")
            return False

    def _thumbnail_from_pil(self, img: Image.Image, photo_id: str) -> str:
        """从 PIL Image 生成缩略图"""
        dest = self.thumb_dir / f"{photo_id}.webp"
        if dest.exists():
            return str(dest)
        img = img.copy()
        img.thumbnail(self.thumb_size)
        img.save(str(dest), "WEBP", quality=85)
        return str(dest)

    def _wait_until_stable(self, path: Path, timeout: float = 60.0, interval: float = 1.0) -> bool:
        """等待文件大小稳定，避免刚写入一半就开始索引。"""
        deadline = time.time() + timeout
        last_size = None
        stable_checks = 0
        while time.time() < deadline:
            if not path.exists() or not path.is_file():
                time.sleep(interval)
                continue
            try:
                size = path.stat().st_size
            except OSError:
                time.sleep(interval)
                continue
            if size > 0 and size == last_size:
                stable_checks += 1
                if stable_checks >= 2:
                    return True
            else:
                stable_checks = 0
                last_size = size
            time.sleep(interval)
        return False

    def _pipeline(self, items, prepare_fn, batch_size, stats, is_video, desc):
        """双端流水线：固定数量 CPU worker 生产 → 有界队列 → GPU 消费线程批量编码。

        CPU 和 GPU 真正并行工作，队列满时自动背压 CPU。
        """
        global _shutdown
        q = queue.Queue(maxsize=batch_size * 3)  # 有界队列，防内存爆炸
        work_q = queue.Queue()
        producer_error = []
        worker_count = min(self.workers, len(items))

        for item in items:
            work_q.put(item)

        def _producer():
            """CPU 线程池：固定 worker 拉取任务，结果塞入有界队列。"""
            workers = []

            def _worker():
                while not _shutdown:
                    try:
                        item = work_q.get_nowait()
                    except queue.Empty:
                        return

                    try:
                        result = prepare_fn(item)
                    except Exception as e:
                        result = {
                            "ok": False,
                            "path": str(item),
                            "error": str(e),
                        }
                    finally:
                        work_q.task_done()

                    q.put(result)

            try:
                for _ in range(worker_count):
                    thread = threading.Thread(target=_worker, daemon=True)
                    workers.append(thread)
                    thread.start()

                for thread in workers:
                    thread.join()
            except Exception as e:
                producer_error.append(e)
            finally:
                q.put(None)  # sentinel：告诉消费者生产完毕

        def _consumer():
            """GPU 消费线程：从队列取 batch，编码 + 写库"""
            batch_imgs = []
            batch_items = []
            while True:
                try:
                    result = q.get(timeout=1)
                except queue.Empty:
                    continue
                if result is None:  # sentinel
                    break
                pbar.update(1)
                if not result["ok"]:
                    stats["failed"] += 1
                else:
                    batch_imgs.append(result["img"])
                    batch_items.append(result)
                # 攒够一批送 GPU
                if len(batch_imgs) >= batch_size:
                    self._encode_and_store_batch(batch_imgs, batch_items, stats, is_video=is_video)
                    pbar.set_postfix_str(f"✓{stats['indexed']} ✗{stats['failed']}")
                    batch_imgs = []
                    batch_items = []
            # 最后一批
            if batch_imgs:
                self._encode_and_store_batch(batch_imgs, batch_items, stats, is_video=is_video)
                pbar.set_postfix_str(f"✓{stats['indexed']} ✗{stats['failed']}")

        pbar = tqdm(total=len(items), desc=desc, unit="张" if not is_video else "个",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

        prod_thread = threading.Thread(target=_producer, daemon=True)
        cons_thread = threading.Thread(target=_consumer, daemon=True)
        prod_thread.start()
        cons_thread.start()
        prod_thread.join()
        cons_thread.join()

        pbar.set_postfix_str(f"✓{stats['indexed']} ✗{stats['failed']}")
        pbar.close()

        if producer_error:
            logger.error(f"生产者异常: {producer_error[0]}")

    def scan_dir(self, dir_path: str) -> dict:
        """扫描目录，返回 {indexed, skipped, failed}。CPU/GPU 双端流水线并行。"""
        global _shutdown
        _shutdown = False

        path = Path(dir_path).expanduser()
        if not path.exists():
            print(f"❌ 目录不存在: {dir_path}")
            return {"indexed": 0, "skipped": 0, "failed": 0}

        print(f"\n🔎 扫描目录: {path}")
        photo_files = sorted(
            f for f in path.rglob("*")
            if f.suffix.lower() in self.exts and f.is_file()
        )
        video_files = sorted(
            f for f in path.rglob("*")
            if f.suffix.lower() in self.video_exts and f.is_file()
        )
        print(f"   发现 {len(photo_files)} 张图片, {len(video_files)} 个视频")

        total = len(photo_files) + len(video_files)
        if total == 0:
            return {"indexed": 0, "skipped": 0, "failed": 0}

        indexed_paths = self.store.get_indexed_paths()
        print(f"   已索引: {len(indexed_paths)} 条")

        stats = {"indexed": 0, "skipped": 0, "failed": 0}
        BATCH_SIZE = 8  # GPU 批量编码大小

        # ---- 阶段 1: 照片双端流水线 ----
        new_photos = [f for f in photo_files if str(f.resolve()) not in indexed_paths]
        stats["skipped"] += len(photo_files) - len(new_photos)

        if new_photos:
            print(f"   双端流水线处理 {len(new_photos)} 张新照片 (batch={BATCH_SIZE})...")

            def _prepare_photo(f):
                """CPU 线程：读图 + EXIF + 缩略图"""
                abs_path = str(f.resolve())
                try:
                    meta = exif.extract(abs_path)
                    photo_id = str(uuid.uuid4())
                    thumb = self._thumbnail(abs_path, photo_id)
                    img = ImageOps.exif_transpose(Image.open(abs_path)).convert("RGB")
                    return {"ok": True, "path": abs_path, "id": photo_id,
                            "meta": meta, "thumb": thumb, "img": img}
                except Exception as e:
                    return {"ok": False, "path": abs_path, "error": str(e)}

            self._pipeline(new_photos, _prepare_photo, BATCH_SIZE, stats,
                           is_video=False, desc="索引照片")

            if _shutdown:
                print(f"\n⚠️  中断退出 — 已保存 {stats['indexed']} 条索引")
                return stats

        # ---- 阶段 2: 视频双端流水线 ----
        new_videos = [f for f in video_files if str(f.resolve()) not in indexed_paths]
        stats["skipped"] += len(video_files) - len(new_videos)

        if new_videos:
            print(f"   双端流水线处理 {len(new_videos)} 个视频...")

            from indexer.video import process_video

            def _prepare_video(f):
                """CPU 线程：FFmpeg 采帧 + 人脸选最佳帧"""
                abs_path = str(f.resolve())
                try:
                    result = process_video(abs_path)
                    if not result:
                        return {"ok": False, "path": abs_path, "error": "无法处理"}
                    photo_id = str(uuid.uuid4())
                    thumb = self._thumbnail_from_pil(result["image"], photo_id)
                    meta = exif.extract(abs_path)
                    return {"ok": True, "path": abs_path, "id": photo_id,
                            "meta": meta, "thumb": thumb, "img": result["image"],
                            "duration": result["duration"], "info": result["info"]}
                except Exception as e:
                    return {"ok": False, "path": abs_path, "error": str(e)}

            self._pipeline(new_videos, _prepare_video, BATCH_SIZE, stats,
                           is_video=True, desc="索引视频")

        if _shutdown:
            print(f"\n⚠️  中断退出 — 已保存 {stats['indexed']} 条索引")
        else:
            print(f"   ✅ 完成: 新索引 {stats['indexed']}, 跳过 {stats['skipped']}, 失败 {stats['failed']}")

        return stats

    def _encode_and_store_batch(self, imgs, items, stats, is_video=False):
        """GPU 批量编码 + 写入数据库"""
        if not imgs:
            return
        try:
            vectors = self.encoder.encode_images(imgs, batch_size=len(imgs))
            for item, vec in zip(items, vectors):
                meta = item["meta"]
                city, country = None, None
                if meta.get("gps_lat") is not None and meta.get("gps_lon") is not None:
                    city, country = geocoder.reverse(meta["gps_lat"], meta["gps_lon"])

                record = {
                    "id": item["id"],
                    "file_path": item["path"],
                    "file_hash": self._hash(item["path"]),
                    "thumbnail_path": item["thumb"],
                    "vector": vec.tolist(),
                    "date_original": meta.get("date_original") or meta.get("date_original") or "",
                    "gps_lat": meta.get("gps_lat"),
                    "gps_lon": meta.get("gps_lon"),
                    "gps_city": city or "",
                    "gps_country": country or "",
                    "camera_make": meta.get("camera_make") or "",
                    "camera_model": meta.get("camera_model") or "",
                    "width": meta.get("width", 0) or item.get("info", {}).get("width", 0),
                    "height": meta.get("height", 0) or item.get("info", {}).get("height", 0),
                    "indexed_at": datetime.now().isoformat(),
                    "media_type": "video" if is_video else "",
                    "video_duration": item.get("duration", 0.0) if is_video else 0.0,
                }
                self.store.add(record)
                stats["indexed"] += 1
        except Exception as e:
            logger.error(f"批量编码失败: {e}")
            stats["failed"] += len(imgs)

    def scan_all(self) -> dict:
        """扫描配置文件中的所有目录"""
        dirs = cfg.get("photo_dirs", [])
        print(f"📋 配置目录: {dirs}")

        total = {"indexed": 0, "skipped": 0, "failed": 0}
        for i, d in enumerate(dirs, 1):
            if _shutdown:
                break
            print(f"\n━━━ 目录 {i}/{len(dirs)} ━━━")
            s = self.scan_dir(d)
            for k in total:
                total[k] += s[k]

        print(f"\n{'='*40}")
        print(f"🏁 全部完成: 新索引 {total['indexed']}, 跳过 {total['skipped']}, 失败 {total['failed']}")
        return total

    def watch(self, dir_path: str):
        """监控目录变化，自动索引新文件"""
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        print("👀 进入监控模式...")

        class Handler(FileSystemEventHandler):
            def __init__(self, indexer):
                self.indexer = indexer

            def _handle_file(self, src_path: str):
                p = Path(src_path)
                suffix = p.suffix.lower()
                if suffix not in self.indexer.exts and suffix not in self.indexer.video_exts:
                    return
                if not self.indexer._wait_until_stable(p):
                    print(f"⚠️  文件未稳定，跳过: {p.name}")
                    return

                print(f"📁 新增: {p.name}")
                if suffix in self.indexer.video_exts:
                    ok = self.indexer.index_video(str(p))
                else:
                    ok = self.indexer.index_file(str(p))
                print(f"   {'✅ 索引成功' if ok else '❌ 索引失败'}")

            def on_created(self, event):
                if not event.is_directory:
                    self._handle_file(event.src_path)

            def on_moved(self, event):
                if not event.is_directory:
                    self._handle_file(event.dest_path)

        path = Path(dir_path).expanduser()
        obs = Observer()
        obs.schedule(Handler(self), str(path), recursive=True)
        obs.start()
        print(f"👀 监控中: {path} (Ctrl+C 退出)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n停止监控")
            obs.stop()
        obs.join()
