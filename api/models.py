"""Pydantic 请求/响应模型"""

from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)


class RefinedSearchRequest(BaseModel):
    """用户调整条件后的搜索"""
    text_query: str = Field(..., min_length=1)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    location: Optional[str] = None
    device: Optional[str] = None
    top_k: int = Field(default=30, ge=1, le=100)


class PhotoResult(BaseModel):
    id: str
    file_path: str
    thumbnail_path: str = ""
    date_original: str = ""
    gps_city: str = ""
    gps_country: str = ""
    camera_make: str = ""
    camera_model: str = ""
    width: int = 0
    height: int = 0
    similarity: float = 0.0


class SearchResponse(BaseModel):
    results: list[PhotoResult]
    total: Optional[int] = None
    has_more: bool = False
    offset: int = 0
    limit: int = 30


class StatsResponse(BaseModel):
    total_photos: int


class IndexRequest(BaseModel):
    path: str


class IndexResponse(BaseModel):
    indexed: int
    skipped: int
    failed: int
