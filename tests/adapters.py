"""
测试适配器：将 cs336_data 中的实现函数连接到测试套件。

每个 run_* 函数是对应实现函数的薄包装，负责：
  1. 调用正确的实现
  2. 处理返回值格式（如语言代码的重映射）
  3. 保持测试接口稳定，即使内部实现变更

使用方式：
  uv run pytest -k test_extract_text_from_html_bytes
  uv run pytest -k test_identify_language
  ...（详见各测试文件）
"""

from __future__ import annotations

import os
from typing import Any


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """从 HTML 字节串提取纯文本。详见 cs336_data/extract.py。"""
    from cs336_data.extract import extract_text_from_html_bytes
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    """语言识别，返回 (language_code, confidence)。

    language_code 已做标准化：
      - 中文繁简体均映射为 "zh"
      - 英文为 "en"
    详见 cs336_data/language_id.py。
    """
    from cs336_data.language_id import identify_language
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    """屏蔽邮件地址，返回 (替换后文本, 替换次数)。详见 cs336_data/pii.py。"""
    from cs336_data.pii import mask_emails
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    """屏蔽电话号码，返回 (替换后文本, 替换次数)。详见 cs336_data/pii.py。"""
    from cs336_data.pii import mask_phone_numbers
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    """屏蔽 IPv4 地址，返回 (替换后文本, 替换次数)。详见 cs336_data/pii.py。"""
    from cs336_data.pii import mask_ips
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    """NSFW 内容分类，返回 ("nsfw"/"non-nsfw", confidence)。
    详见 cs336_data/harmful.py。
    """
    from cs336_data.harmful import classify_nsfw
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    """有毒言论分类，返回 ("toxic"/"non-toxic", confidence)。
    详见 cs336_data/harmful.py。
    """
    from cs336_data.harmful import classify_toxic_speech
    return classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    """质量分类，返回 ("wiki"/"cc", confidence)。

    注意：需要先运行训练脚本生成模型：
      uv run python extra_guidance/train_quality_classifier.py
    详见 cs336_data/quality.py。
    """
    from cs336_data.quality import classify_quality
    return classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    """Gopher 质量过滤，True = 通过（保留）。详见 cs336_data/quality.py。"""
    from cs336_data.quality import gopher_quality_filter
    return gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """精确行去重。详见 cs336_data/dedup.py。"""
    from cs336_data.dedup import exact_line_deduplication
    exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """MinHash + LSH 模糊文档去重。详见 cs336_data/dedup.py。"""
    from cs336_data.dedup import minhash_deduplication
    minhash_deduplication(
        input_files=input_files,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
        output_directory=output_directory,
    )
