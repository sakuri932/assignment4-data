"""
语言模型训练数据过滤流水线（Problem filter_data）。

目标：从 /shared-data/english-wet-data/ 中的 CC WET 文件出发，
      生成能最小化 Paloma C4-100-domains 验证困惑度的训练数据。

过滤步骤（按执行顺序）：
  1. 文本提取：使用 Resiliparse 从 WET 记录中读取文本
     （WET 文件已是纯文本，但仍有 HTTP 头部需跳过）
  2. Gopher 质量过滤：词数/词长/省略号/字母占比四项规则
  3. 语言确认：fastText lid.176.bin，英文置信度 >= 0.65
     （WET 文件已预过滤 >= 0.7，这里再做一次防漏）
  4. NSFW 过滤：Dolma fastText 分类器，score < 0.5 为非 NSFW
  5. 仇恨言论过滤：Dolma fastText 分类器，score < 0.5 为非有毒
  6. 质量分类器：fastText wiki/cc 分类器，保留 "wiki" 或置信度 > 阈值
  7. PII 屏蔽：替换邮件/电话/IP

并行处理：使用 concurrent.futures.ProcessPoolExecutor

运行方式：
  uv run python extra_guidance/filter_pipeline.py \
      --input-dir /shared-data/english-wet-data \
      --output-dir /mnt/a/kong/workspace/ass4/filtered \
      --workers 8
"""

import argparse
import concurrent.futures
import gzip
import logging
import os
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 过滤阈值（可通过 Paloma 验证集调整）
LANG_THRESHOLD = 0.65           # 语言识别置信度（英文）
NSFW_THRESHOLD = 0.5            # NSFW 置信度上限（超过则丢弃）
HATE_THRESHOLD = 0.5            # 仇恨言论置信度上限
QUALITY_THRESHOLD = 0.5         # 质量分类器置信度下限（"wiki" 预测时）
USE_QUALITY_CLASSIFIER = True   # 是否启用质量分类器（需先训练）


def process_single_wet_file(args: tuple[str, str]) -> dict:
    """处理单个 WET 文件，返回过滤统计信息。

    设计为顶层函数（非嵌套）以支持 multiprocessing pickle。

    Args:
        args: (input_path, output_path) 元组。

    Returns:
        包含各步骤统计的字典。
    """
    input_path, output_path = args

    # 延迟导入（子进程中初始化，避免 fork 问题）
    from cs336_data.harmful import classify_nsfw, classify_toxic_speech
    from cs336_data.language_id import identify_language
    from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
    from cs336_data.quality import classify_quality, gopher_quality_filter

    stats = {
        "total": 0,
        "kept": 0,
        "dropped_gopher": 0,
        "dropped_lang": 0,
        "dropped_nsfw": 0,
        "dropped_hate": 0,
        "dropped_quality": 0,
    }

    kept_texts = []

    try:
        with gzip.open(input_path, "rb") as f:
            for rec in ArchiveIterator(f):
                # WET 文件只有 conversion（纯文本）和 warcinfo 两种记录
                if rec.record_type != WarcRecordType.conversion:
                    continue

                payload = rec.reader.read()
                text = payload.decode("utf-8", errors="replace")
                stats["total"] += 1

                # ── 1. Gopher 质量过滤 ────────────────────────────────────
                if not gopher_quality_filter(text):
                    stats["dropped_gopher"] += 1
                    continue

                # ── 2. 语言确认（英文） ───────────────────────────────────
                lang, lang_score = identify_language(text)
                if lang != "en" or lang_score < LANG_THRESHOLD:
                    stats["dropped_lang"] += 1
                    continue

                # ── 3. NSFW 过滤 ─────────────────────────────────────────
                nsfw_label, nsfw_score = classify_nsfw(text)
                if nsfw_label == "nsfw" and nsfw_score > NSFW_THRESHOLD:
                    stats["dropped_nsfw"] += 1
                    continue

                # ── 4. 仇恨言论过滤 ──────────────────────────────────────
                hate_label, hate_score = classify_toxic_speech(text)
                if hate_label == "toxic" and hate_score > HATE_THRESHOLD:
                    stats["dropped_hate"] += 1
                    continue

                # ── 5. 质量分类器 ─────────────────────────────────────────
                if USE_QUALITY_CLASSIFIER:
                    try:
                        q_label, q_score = classify_quality(text)
                        if q_label != "wiki" or q_score < QUALITY_THRESHOLD:
                            stats["dropped_quality"] += 1
                            continue
                    except FileNotFoundError:
                        pass  # 如果质量分类器未训练，跳过该步骤

                # ── 6. PII 屏蔽 ────────────────────────────────────────────
                text, _ = mask_emails(text)
                text, _ = mask_phone_numbers(text)
                text, _ = mask_ips(text)

                kept_texts.append(text)
                stats["kept"] += 1

    except Exception as e:
        logger.error("处理 %s 时出错: %s", input_path, e)
        return stats

    # 写出过滤后文本（每篇文档一段，段间空行分隔）
    if kept_texts:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for text in kept_texts:
                f.write(text.strip())
                f.write("\n\n")

    return stats


def run_pipeline(input_dir: str, output_dir: str, workers: int = None) -> None:
    """运行完整的过滤流水线。"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wet_files = sorted(input_path.glob("*.warc.wet.gz"))
    if not wet_files:
        logger.error("在 %s 中未找到 WET 文件", input_dir)
        return

    logger.info("发现 %d 个 WET 文件，开始处理...", len(wet_files))

    if workers is None:
        workers = len(os.sched_getaffinity(0))

    # 构建任务列表
    tasks = []
    for wet_file in wet_files:
        out_file = output_path / (wet_file.stem.replace(".warc.wet", "") + ".txt")
        tasks.append((str(wet_file), str(out_file)))

    # 并行处理
    total_stats = {
        "total": 0, "kept": 0,
        "dropped_gopher": 0, "dropped_lang": 0,
        "dropped_nsfw": 0, "dropped_hate": 0, "dropped_quality": 0,
    }

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_wet_file, task): task for task in tasks}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            stats = future.result()
            for k in total_stats:
                total_stats[k] += stats[k]
            if completed % 50 == 0:
                logger.info("已处理 %d/%d 个文件", completed, len(tasks))

    logger.info("=" * 60)
    logger.info("过滤完成。统计：")
    logger.info("  总文档数:               %8d (100%%)", total_stats["total"])
    logger.info("  → Gopher 过滤丢弃:      %8d (%4.1f%%)",
                total_stats["dropped_gopher"],
                100 * total_stats["dropped_gopher"] / max(total_stats["total"], 1))
    logger.info("  → 语言过滤丢弃:         %8d (%4.1f%%)",
                total_stats["dropped_lang"],
                100 * total_stats["dropped_lang"] / max(total_stats["total"], 1))
    logger.info("  → NSFW 过滤丢弃:        %8d (%4.1f%%)",
                total_stats["dropped_nsfw"],
                100 * total_stats["dropped_nsfw"] / max(total_stats["total"], 1))
    logger.info("  → 仇恨言论过滤丢弃:     %8d (%4.1f%%)",
                total_stats["dropped_hate"],
                100 * total_stats["dropped_hate"] / max(total_stats["total"], 1))
    logger.info("  → 质量分类器丢弃:       %8d (%4.1f%%)",
                total_stats["dropped_quality"],
                100 * total_stats["dropped_quality"] / max(total_stats["total"], 1))
    logger.info("  最终保留:               %8d (%4.1f%%)",
                total_stats["kept"],
                100 * total_stats["kept"] / max(total_stats["total"], 1))
    logger.info("输出目录: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="CC WET 文件过滤流水线")
    parser.add_argument("--input-dir", required=True, help="英文 WET 文件目录")
    parser.add_argument("--output-dir", required=True, help="过滤后文本输出目录")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数（默认 CPU 核数）")
    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
