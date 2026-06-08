"""Web 服务入口"""

import shared.compat  # noqa: F401 - 必须最先导入（torch 兼容补丁）

import logging
import uvicorn
import shared.config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

cfg.load("config.yaml")
port = cfg.get("port", 8080)

uvicorn.run("api.app:app", host="0.0.0.0", port=port, reload=False)
