"""索引服务入口"""

import shared.compat  # noqa: F401 - 必须最先导入（torch 兼容补丁）

import sys
import logging
import argparse

from tqdm import tqdm


class _TqdmLogHandler(logging.StreamHandler):
    """让 logging 输出通过 tqdm.write，避免和进度条混在一起"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[_TqdmLogHandler()],
)

import shared.config as cfg


def main():
    ap = argparse.ArgumentParser(description="Photo Search - 索引服务")
    ap.add_argument("--config", default="config.yaml", help="配置文件路径")
    ap.add_argument("--scan-all", action="store_true", help="扫描配置中的所有目录")
    ap.add_argument("--watch", action="store_true", help="监控模式 (需指定目录)")
    ap.add_argument("directory", nargs="?", help="要索引的目录")
    args = ap.parse_args()

    cfg.load(args.config)

    from indexer.indexer import Indexer
    idx = Indexer()

    if args.scan_all:
        stats = idx.scan_all()
    elif args.directory:
        if args.watch:
            idx.watch(args.directory)
        else:
            stats = idx.scan_dir(args.directory)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
