"""FastAPI 应用"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import shared.config as cfg
from api import search as search_mod
from api.face_api import router as face_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("服务启动")
    yield


app = FastAPI(title="Photo Search", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 请求模型 ----

class SearchRequest(BaseModel):
    query: str = ""
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    location: Optional[str] = None
    device: Optional[str] = None
    top_k: int = 30
    offset: int = 0


# ---- API 路由 ----

@app.post("/api/search")
async def do_search(req: SearchRequest):
    try:
        result = search_mod.search(
            text_query=req.query,
            date_from=req.date_from,
            date_to=req.date_to,
            location=req.location,
            device=req.device,
            top_k=req.top_k,
            offset=req.offset,
        )
        return result
    except Exception as e:
        logger.exception("搜索失败")
        raise HTTPException(500, str(e))


@app.get("/api/photos/{photo_id}/thumbnail")
async def get_thumbnail(photo_id: str):
    thumb_dir = Path(cfg.get("data_dir", "./data")) / "thumbnails"
    path = thumb_dir / f"{photo_id}.webp"
    if path.exists():
        return FileResponse(str(path), media_type="image/webp")
    raise HTTPException(404, "缩略图不存在")


@app.get("/api/faces/{face_id}/thumbnail")
async def get_face_thumbnail(face_id: str):
    """人脸缩略图"""
    face_dir = Path(cfg.get("data_dir", "./data")) / "faces"
    path = face_dir / f"{face_id}.webp"
    if path.exists():
        return FileResponse(str(path), media_type="image/webp")
    raise HTTPException(404, "人脸缩略图不存在")


@app.get("/api/photos/{photo_id}/file")
async def get_photo_file(photo_id: str):
    """返回原图"""
    from shared.store import Store
    store = Store(cfg.get("data_dir", "./data"))
    try:
        df = store.table.to_pandas(filter=f"id = '{photo_id}'")
        if df.empty:
            raise HTTPException(404, "照片不存在")
        file_path = df.iloc[0]["file_path"]
        if not Path(file_path).exists():
            raise HTTPException(404, "文件不存在")
        return FileResponse(str(file_path))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/stats")
async def get_stats():
    from shared.store import Store
    store = Store(cfg.get("data_dir", "./data"))
    return {"total_photos": store.count()}


@app.post("/api/index")
async def trigger_index(req: dict):
    from indexer.indexer import Indexer
    idx = Indexer()
    try:
        stats = idx.scan_dir(req.get("path", ""))
        return stats
    except Exception as e:
        raise HTTPException(500, str(e))


# ---- 人脸识别路由 ----
app.include_router(face_router)


# ---- 前端静态文件 ----
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
