"""
分词与序列化脚本（Problem tokenize_data）。

将过滤后的文本文件（每篇文档一段）用 GPT-2 分词器编码为 uint16 整数序列，
序列化为二进制文件，供训练脚本直接读取。

格式要求：
  - 每篇文档末尾添加 EOS token（GPT-2 的 <|endoftext|> = 50256）
  - 以 np.uint16 格式保存（GPT-2 词表 50257 个，uint16 足够）
  - 文件以 ids_array.tofile(output_path) 形式写出

运行方式：
  uv run python extra_guidance/tokenize_data.py \
      --input-dir /path/to/filtered \
      --output-path /path/to/data.bin
"""

import argparse
import logging
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# GPT-2 分词器（词表大小 50257，uint16 可容纳）
_TOKENIZER = None


def _init_tokenizer():
    """子进程初始化函数：加载 GPT-2 分词器。"""
    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def tokenize_document(text: str) -> list[int]:
    """对单篇文档分词并添加 EOS token。"""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
    return _TOKENIZER.encode(text) + [_TOKENIZER.eos_token_id]


def tokenize_file(file_path: Path) -> list[int]:
    """对整个文件中的所有文档分词（文档间以双空行分隔）。"""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    # 按双空行切分文档
    documents = [doc.strip() for doc in text.split("\n\n") if doc.strip()]
    all_ids = []
    for doc in documents:
        all_ids.extend(tokenize_document(doc))
    return all_ids


def main():
    parser = argparse.ArgumentParser(description="将过滤后文本分词序列化")
    parser.add_argument("--input-dir", required=True, help="过滤后文本目录")
    parser.add_argument("--output-path", required=True, help="输出 .bin 文件路径")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help="并行进程数")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        logger.error("在 %s 中未找到 .txt 文件", input_dir)
        return

    logger.info("发现 %d 个文件，开始分词...", len(txt_files))

    # 使用多进程并行分词
    all_ids = []
    with multiprocessing.Pool(
        processes=args.workers,
        initializer=_init_tokenizer
    ) as pool:
        for ids in tqdm(
            pool.imap(tokenize_file, txt_files, chunksize=10),
            total=len(txt_files),
            desc="分词进度"
        ):
            all_ids.extend(ids)

    logger.info("总 token 数量: %d", len(all_ids))
    logger.info("估算存储大小: %.2f GB（uint16）", len(all_ids) * 2 / 1e9)

    # 序列化为 uint16 二进制文件
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(str(output_path))
    logger.info("已写出: %s（%d tokens）", output_path, len(all_ids))


if __name__ == "__main__":
    main()
