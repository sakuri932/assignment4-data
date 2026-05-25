"""
在训练服务器上批量下载 Common Crawl WET 文件。

WET 文件（WARC-Extracted-Text）是 Common Crawl 的纯文本版，
每个文件约 60-70 MB（压缩），包含约 50K-200K 条网页文本记录。

下载后直接运行 extra_guidance/filter_pipeline.py 进行过滤。

运行方式：
  python extra_guidance/download_wet_files.py \
      --output-dir data/wet \
      --n-files 200 \
      --workers 4

目录结构（推荐）：
  ass4/
    data/
      wet/         ← 本脚本输出（原始 WET 压缩包）
      filtered/    ← filter_pipeline.py 输出
      data.bin     ← tokenize_data.py 输出
"""

import argparse
import gzip
import logging
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CC_BASE_URL = "https://data.commoncrawl.org"
CC_CRAWL_ID = "CC-MAIN-2026-17"
USER_AGENT = "CS336DataPipeline/1.0"


def fetch_wet_paths(crawl_id: str) -> list[str]:
    url = f"{CC_BASE_URL}/crawl-data/{crawl_id}/wet.paths.gz"
    logger.info("获取 WET 路径列表: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    paths = gzip.decompress(raw).decode().strip().split("\n")
    logger.info("共 %d 个 WET 文件", len(paths))
    return paths


def download_one(path: str, output_dir: Path) -> bool:
    filename = Path(path).name
    dest = output_dir / filename
    if dest.exists():
        logger.info("已存在，跳过: %s", filename)
        return True
    url = f"{CC_BASE_URL}/{path}"
    tmp = dest.with_suffix(".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            while chunk := resp.read(65536):
                f.write(chunk)
        tmp.rename(dest)
        logger.info("完成: %s (%.1f MB)", filename, dest.stat().st_size / 1024 / 1024)
        return True
    except Exception as e:
        logger.warning("失败: %s — %s", filename, e)
        tmp.unlink(missing_ok=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="批量下载 CC WET 文件")
    parser.add_argument("--output-dir", default="data/wet", help="保存目录（默认 data/wet）")
    parser.add_argument("--n-files", type=int, default=200, help="下载数量（默认 200）")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数（默认 4）")
    parser.add_argument("--seed", type=int, default=336, help="随机种子")
    parser.add_argument("--crawl-id", default=CC_CRAWL_ID, help="CC 爬取 ID")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = fetch_wet_paths(args.crawl_id)
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    selected = paths[: args.n_files]

    logger.info("将下载 %d 个 WET 文件到 %s", len(selected), output_dir)
    logger.info("预计总大小: ~%.0f GB（压缩）", len(selected) * 64 / 1024)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, p, output_dir): p for p in selected}
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                ok += 1
            else:
                fail += 1
            if i % 10 == 0:
                logger.info("进度 %d/%d（成功 %d，失败 %d）", i, len(selected), ok, fail)

    logger.info("下载完成：成功 %d，失败 %d，输出目录 %s", ok, fail, output_dir)
    logger.info("下一步：")
    logger.info("  python extra_guidance/filter_pipeline.py \\")
    logger.info("      --input-dir %s \\", output_dir)
    logger.info("      --output-dir data/filtered \\")
    logger.info("      --workers 32")


if __name__ == "__main__":
    main()
