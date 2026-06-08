"""SigLIP2 编码器"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

import shared.config as cfg

logger = logging.getLogger(__name__)

_encoder = None


def get_encoder():
    """单例获取编码器"""
    global _encoder
    if _encoder is None:
        _encoder = _Encoder()
    return _encoder


class _Encoder:
    def __init__(self):
        model_path = cfg.get("models.siglip2")
        device = cfg.resolve_device()
        logger.info(f"加载 SigLIP2: {model_path} ({device})")

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        logger.info("SigLIP2 就绪")

    @torch.no_grad()
    def encode_image(self, image) -> np.ndarray:
        """图片 → 1152维向量 (已归一化)"""
        if isinstance(image, (str, Path)):
            image = ImageOps.exif_transpose(Image.open(image)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        vec = out.cpu().numpy().flatten()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @torch.no_grad()
    def encode_images(self, images: list, batch_size: int = 16) -> np.ndarray:
        """批量编码图片"""
        all_vecs = []
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(ImageOps.exif_transpose(Image.open(img)).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))

        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            out = self.model.get_image_features(**inputs)
            if hasattr(out, "pooler_output"):
                out = out.pooler_output
            vecs = out.cpu().numpy()
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            all_vecs.append(vecs / norms)

        return np.vstack(all_vecs)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """文本 → 1152维向量 (已归一化)"""
        inputs = self.processor(
            text=text, return_tensors="pt",
            padding="max_length", max_length=64,
        ).to(self.device)
        out = self.model.get_text_features(**inputs)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        vec = out.cpu().numpy().flatten()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
