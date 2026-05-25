"""
质量分类器训练脚本。

核心思想（来自 CCNet、GPT-2 WebText、RefinedWeb 等工作）：
  - 正例（high-quality wiki）：被英文维基百科外链引用的页面
    → 维基百科编辑社区有严格的引用质量标准，外链通常指向可信、信息丰富的来源
  - 负例（low-quality cc）：从 Common Crawl 随机抽取的页面
    → 代表原始互联网的平均质量水平

训练一个 fastText 二分类器，标签：
  __label__wiki → 高质量
  __label__cc   → 低质量

输出：local-shared-data/classifiers/quality_classifier.bin

运行方式：
  uv run python extra_guidance/train_quality_classifier.py

前置条件：
  - 英文 WET 文件在 /shared-data/english-wet-data/ 或 local-shared-data/english-wet-data/
  - 维基百科 URL 文件在 /shared-data/wiki/enwiki-20260501-extracted_urls.txt.gz 或本地
  - wget 已安装（用于抓取维基百科引用的页面）

注意：
  - 可以使用 Paloma 验证集数据辅助设计过滤器，但不能将验证集数据放入训练集
"""

import gzip
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.common import get_shared_assets_path
from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.quality import gopher_quality_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 超参数
N_POSITIVE_URLS = 10_000       # 从维基百科外链中采样的 URL 数量
N_NEGATIVE_DOCS = 10_000       # 从 CC WET 中采样的负例数量
FASTTEXT_DIM = 64              # 词向量维度（小模型足够区分两类）
FASTTEXT_EPOCHS = 5
FASTTEXT_LR = 0.5
FASTTEXT_WORDNGRAMS = 2        # 使用 bigram，与 Dolma 的 Jigsaw 模型一致
MIN_TEXT_LEN = 100             # 过滤过短文本


def load_wiki_urls(n: int = N_POSITIVE_URLS) -> list[str]:
    """从维基百科外链文件中随机采样 URL。"""
    shared = get_shared_assets_path()
    url_file = shared / "wiki" / "enwiki-20260501-extracted_urls.txt.gz"
    if not url_file.exists():
        raise FileNotFoundError(f"维基百科 URL 文件不存在: {url_file}")

    logger.info("读取维基百科外链 URL 文件: %s", url_file)
    urls = []
    with gzip.open(url_file, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            url = line.strip()
            if url and url.startswith("http"):
                urls.append(url)

    rng = random.Random(336)
    sampled = rng.sample(urls, min(n, len(urls)))
    logger.info("采样了 %d 个 URL（共 %d 个）", len(sampled), len(urls))
    return sampled


def scrape_urls_to_warc(urls: list[str], output_warc: Path) -> None:
    """使用 wget 将 URL 列表抓取为 WARC 格式。

    等价于 PDF 中的示例命令：
      wget --timeout=5 -i urls.txt --warc-file=output.warc -O /dev/null
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(urls))
        url_file = f.name

    warc_prefix = str(output_warc).replace(".warc.gz", "").replace(".warc", "")
    cmd = [
        "wget",
        "--timeout=5",
        "--tries=1",
        "-q",
        "-i", url_file,
        f"--warc-file={warc_prefix}",
        "-O", "/dev/null",
    ]
    logger.info("开始抓取 %d 个 URL...", len(urls))
    subprocess.run(cmd, check=False)  # 允许部分失败（超时、404 等）
    os.unlink(url_file)
    logger.info("WARC 文件生成: %s", output_warc)


def extract_texts_from_warc(warc_path: Path, max_texts: int) -> list[str]:
    """从 WARC 文件中提取纯文本（HTML response 记录）。"""
    texts = []
    try:
        with open(warc_path, "rb") as f:
            for rec in ArchiveIterator(f):
                if len(texts) >= max_texts:
                    break
                # 只处理 HTML 响应记录
                if rec.rec_type not in ("response", "conversion"):
                    continue
                content_type = rec.http_headers.get("Content-Type", "") if rec.http_headers else ""
                if "html" not in content_type.lower() and rec.rec_type != "conversion":
                    continue
                payload = rec.reader.read()
                if rec.rec_type == "conversion":
                    text = payload.decode("utf-8", errors="replace")
                else:
                    text = extract_text_from_html_bytes(payload) or ""
                if len(text) >= MIN_TEXT_LEN:
                    texts.append(text)
    except Exception as e:
        logger.warning("处理 WARC 文件时出错: %s", e)
    return texts


def load_cc_negative_texts(n: int = N_NEGATIVE_DOCS) -> list[str]:
    """从英文 WET 文件中随机采样负例文本。"""
    shared = get_shared_assets_path()
    wet_dir = shared / "english-wet-data"
    if not wet_dir.exists():
        raise FileNotFoundError(f"英文 WET 数据目录不存在: {wet_dir}")

    wet_files = sorted(wet_dir.glob("*.warc.wet.gz"))
    if not wet_files:
        raise FileNotFoundError(f"在 {wet_dir} 中未找到 .warc.wet.gz 文件")

    rng = random.Random(336)
    rng.shuffle(wet_files)

    texts = []
    for wet_file in wet_files:
        if len(texts) >= n:
            break
        try:
            with gzip.open(wet_file, "rb") as f:
                for rec in ArchiveIterator(f):
                    if len(texts) >= n:
                        break
                    if rec.rec_type != "conversion":
                        continue
                    payload = rec.reader.read()
                    text = payload.decode("utf-8", errors="replace")
                    if len(text) >= MIN_TEXT_LEN:
                        texts.append(text)
        except Exception as e:
            logger.warning("读取 %s 时出错: %s", wet_file, e)

    logger.info("从 CC 中加载了 %d 条负例文本", len(texts))
    return texts


def write_training_file(
    positive_texts: list[str],
    negative_texts: list[str],
    output_path: Path,
) -> None:
    """写出 fastText 训练文件（每行一个样本，格式：__label__xxx text）。

    对文本做基础清洗：去掉换行符（fastText 每行 = 一个样本）。
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for text in positive_texts:
            line = text.replace("\n", " ").strip()
            if line:
                f.write(f"__label__wiki {line}\n")
        for text in negative_texts:
            line = text.replace("\n", " ").strip()
            if line:
                f.write(f"__label__cc {line}\n")
    logger.info(
        "训练文件写出: %s（正例 %d，负例 %d）",
        output_path, len(positive_texts), len(negative_texts)
    )


def train_classifier(train_file: Path, model_output: Path) -> None:
    """用 fastText 训练二分类器。"""
    model = fasttext.train_supervised(
        input=str(train_file),
        dim=FASTTEXT_DIM,
        epoch=FASTTEXT_EPOCHS,
        lr=FASTTEXT_LR,
        wordNgrams=FASTTEXT_WORDNGRAMS,
        loss="softmax",
        verbose=2,
    )
    model.save_model(str(model_output))
    logger.info("模型保存至: %s", model_output)

    # 打印训练集上的简单精度（用于调试）
    result = model.test(str(train_file))
    logger.info(
        "训练集评估：样本数 %d，精确率 %.4f，召回率 %.4f",
        result[0], result[1], result[2]
    )


def main() -> None:
    shared = get_shared_assets_path()
    classifiers_dir = shared / "classifiers"
    classifiers_dir.mkdir(parents=True, exist_ok=True)

    model_path = classifiers_dir / "quality_classifier.bin"
    if model_path.exists():
        logger.info("模型已存在: %s，跳过训练", model_path)
        return

    # ── Step 1：获取正例（维基百科引用的页面）─────────────────────────────
    logger.info("Step 1: 加载维基百科外链 URL")
    urls = load_wiki_urls(N_POSITIVE_URLS)

    with tempfile.TemporaryDirectory() as tmpdir:
        warc_path = Path(tmpdir) / "wiki_positive.warc"
        logger.info("Step 2: 抓取正例页面（WARC 格式）")
        scrape_urls_to_warc(urls, warc_path)

        # wget 生成的文件可能是 .warc.gz 格式
        if not warc_path.exists():
            warc_path = Path(str(warc_path) + ".gz")

        logger.info("Step 3: 从 WARC 中提取正例文本")
        positive_texts = extract_texts_from_warc(warc_path, N_POSITIVE_URLS)

    # 对正例也应用 Gopher 过滤（去掉明显低质量的引用页面）
    positive_texts = [t for t in positive_texts if gopher_quality_filter(t)]
    logger.info("Gopher 过滤后正例数量: %d", len(positive_texts))

    # ── Step 2：获取负例（CC 随机页面）──────────────────────────────────
    logger.info("Step 4: 加载 CC 负例文本")
    negative_texts = load_cc_negative_texts(N_NEGATIVE_DOCS)

    if not positive_texts:
        raise RuntimeError("未能获取到正例文本，请检查维基百科 URL 和 wget 是否正常工作")

    # ── Step 3：训练分类器 ────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = Path(tmpdir) / "quality_train.txt"
        write_training_file(positive_texts, negative_texts, train_file)
        logger.info("Step 5: 训练 fastText 分类器")
        train_classifier(train_file, model_path)

    logger.info("训练完成！模型已保存到: %s", model_path)


if __name__ == "__main__":
    main()
