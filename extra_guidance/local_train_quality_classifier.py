"""
本地质量分类器训练脚本（无需 Stanford 服务器）。

正例来源：维基百科文章全文（Wikipedia Action API，批量随机获取）
  - 维基百科文章经编辑社区严格审核，代表高质量文本
  - 通过 generator=random + prop=extracts 批量拉取，每次最多 20 篇

负例来源：Common Crawl WET 文件（直接从 data.commoncrawl.org 下载）
  - WET 文件是 Common Crawl WARC 文件的纯文本提取版本，代表平均互联网质量
  - 只需下载 1 个 WET 文件（约 150MB），即可采样数千条负例

训练：fastText 二分类器（同 train_quality_classifier.py，标签 __label__wiki / __label__cc）

输出：local-shared-data/classifiers/quality_classifier.bin

运行方式：
  uv run python extra_guidance/local_train_quality_classifier.py

预计时间：
  - Wikipedia 拉取：~3 分钟（50 次 API 调用）
  - CC WET 下载：~5-15 分钟（约 150MB，视网速而定）
  - 训练：~30 秒
"""

import concurrent.futures
import gzip
import json
import logging
import random
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import fasttext
from fastwarc.warc import ArchiveIterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 超参数 ──────────────────────────────────────────────────────────────────
N_POSITIVE = 5000          # 目标正例数量（Wikipedia 精选条目，上限约 6600）
N_NEGATIVE = 10000         # 目标负例数量（CC WET 文本，对齐原脚本设计）
MIN_TEXT_LEN = 200         # 过短文本不使用

FASTTEXT_DIM = 64
FASTTEXT_EPOCHS = 5
FASTTEXT_LR = 0.5
FASTTEXT_WORDNGRAMS = 2    # bigram，与 Dolma 模型一致

# CC-MAIN 爬取 ID（2026 年第 17 周爬取）
CC_CRAWL_ID = "CC-MAIN-2026-17"
CC_BASE_URL = "https://data.commoncrawl.org"

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/summary"
WIKI_UA = "CS336QualityClassifier/1.0 (kousakuri932@gmail.com)"
WIKI_WORKERS = 10   # 并发线程数（REST API 对内容读取比较宽松）


# ── 正例：Wikipedia API ──────────────────────────────────────────────────────

def _wiki_request(params: dict) -> dict:
    """向 Wikipedia Action API 发起 GET 请求，自动添加 User-Agent 与重试。"""
    params = {**params, "format": "json", "formatversion": "2"}
    query = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
    url = f"{WIKI_API}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.load(resp)
        except (urllib.error.URLError, OSError) as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return {}


def _fetch_summary(title: str) -> str:
    """通过 REST API 获取单篇文章的摘要段落（无 exlimit 限制，并发友好）。

    REST summary endpoint 返回文章第一段（intro），通常 200-600 字符，
    是维基百科最精炼的百科文字，适合作为高质量文本的代表。
    """
    url = f"{WIKI_REST}/{urllib.request.quote(title, safe='')}"
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.load(resp)
            return data.get("extract", "")
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):  # 文章不存在或无效标题
                return ""
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, OSError):
            time.sleep(2 ** attempt)
    return ""


def _get_featured_article_titles(n: int) -> list[str]:
    """从 Wikipedia Featured Articles 分类获取标题列表（支持翻页）。

    Featured Articles 约 6600 篇，经同行评审，内容充实。
    每次 API 调用返回 500 个标题，多次翻页直到够 n 个。
    """
    all_titles: list[str] = []
    continue_token: dict = {}

    while len(all_titles) < n:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Featured articles",
            "cmlimit": "500",
            "cmtype": "page",
            **continue_token,
        }
        try:
            data = _wiki_request(params)
        except Exception as e:
            logger.warning("获取精选条目列表失败: %s", e)
            break
        members = data.get("query", {}).get("categorymembers", [])
        all_titles.extend(m["title"] for m in members)
        if "continue" not in data:
            break
        continue_token = data["continue"]
        time.sleep(0.2)

    rng = random.Random(336)
    rng.shuffle(all_titles)
    return all_titles[:n]


def fetch_wiki_articles(n: int = N_POSITIVE) -> list[str]:
    """从 Wikipedia 精选条目获取摘要段落作为正例。

    策略：
      Step A: 批量获取精选条目标题（Action API，1 次调用 500 个）
      Step B: 用 REST API /page/summary 并发拉取摘要文本
              - REST API 无 exlimit 限制，每篇独立请求，WIKI_WORKERS 并发
              - 速度约 10 篇/秒，5000 篇约 8 分钟

    摘要（intro paragraph）是文章最精炼的部分，充分代表维基百科写作风格。
    """
    logger.info("Step A: 获取 Wikipedia 精选条目标题（目标 %d 篇）...", n)
    titles = _get_featured_article_titles(n + 200)   # 多取缓冲，防止部分标题无摘要
    logger.info("获取到 %d 个精选条目标题", len(titles))

    texts: list[str] = []
    batch_size = WIKI_WORKERS * 5  # 每轮提交的任务数

    with concurrent.futures.ThreadPoolExecutor(max_workers=WIKI_WORKERS) as executor:
        idx = 0
        while len(texts) < n and idx < len(titles):
            batch = titles[idx: idx + batch_size]
            idx += batch_size

            futures = {executor.submit(_fetch_summary, t): t for t in batch}
            for future in concurrent.futures.as_completed(futures):
                text = future.result().strip()
                if len(text) >= MIN_TEXT_LEN:
                    texts.append(text)
                if len(texts) >= n:
                    break

            logger.info("已获取 %d / %d 篇 Wikipedia 文章", len(texts), n)

    logger.info("正例采集完成：%d 篇（通过率 %.0f%%）",
                len(texts), 100 * len(texts) / max(idx, 1))
    return texts[:n]


# ── 负例：Common Crawl WET ───────────────────────────────────────────────────

def _download_url(url: str, dest: Path, desc: str = "") -> None:
    """带进度显示的文件下载。"""
    logger.info("下载 %s => %s", url, dest)
    downloaded = 0

    def _reporthook(block, block_size, total):
        nonlocal downloaded
        downloaded = block * block_size
        if total > 0 and downloaded % (10 * 1024 * 1024) < block_size:
            pct = min(100, downloaded * 100 // total)
            logger.info("  进度 %d%% (%d MB / %d MB)", pct,
                        downloaded // 1024 // 1024, total // 1024 // 1024)

    req = urllib.request.Request(url, headers={"User-Agent": WIKI_UA})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        block_size = 65536
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (10 * 1024 * 1024) < block_size:
                pct = min(100, downloaded * 100 // total)
                logger.info("  进度 %d%% (%d MB / %d MB)", pct,
                            downloaded // 1024 // 1024, total // 1024 // 1024)
    logger.info("下载完成: %s (%.1f MB)", dest, dest.stat().st_size / 1024 / 1024)


def fetch_cc_wet_paths(crawl_id: str = CC_CRAWL_ID) -> list[str]:
    """下载并解析 CC WET 路径列表。"""
    url = f"{CC_BASE_URL}/crawl-data/{crawl_id}/wet.paths.gz"
    logger.info("获取 CC WET 路径列表: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    paths = gzip.decompress(raw).decode().strip().split("\n")
    logger.info("共 %d 个 WET 文件路径", len(paths))
    return paths


def fetch_cc_negative_texts(n: int = N_NEGATIVE) -> list[str]:
    """从 Common Crawl WET 文件中采样负例文本。

    只下载 1 个 WET 文件（约 150MB 压缩），采样 n 条英文文本。
    """
    paths = fetch_cc_wet_paths()
    rng = random.Random(336)
    rng.shuffle(paths)

    texts = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for path in paths:
            if len(texts) >= n:
                break
            wet_url = f"{CC_BASE_URL}/{path}"
            wet_local = Path(tmpdir) / "sample.warc.wet.gz"
            try:
                _download_url(wet_url, wet_local)
            except Exception as e:
                logger.warning("下载失败，跳过 %s: %s", path, e)
                continue

            logger.info("从 WET 文件中提取负例文本...")
            try:
                from fastwarc.warc import WarcRecordType
                with gzip.open(wet_local, "rb") as f:
                    for rec in ArchiveIterator(f):
                        if len(texts) >= n:
                            break
                        if rec.record_type != WarcRecordType.conversion:
                            continue
                        payload = rec.reader.read()
                        text = payload.decode("utf-8", errors="replace")
                        if len(text) >= MIN_TEXT_LEN:
                            texts.append(text)
            except Exception as e:
                logger.warning("解析 WET 文件时出错: %s", e)

            logger.info("已采集 %d / %d 条 CC 负例", len(texts), n)

    logger.info("负例采集完成：%d 条", len(texts))
    return texts[:n]


# ── 训练 ─────────────────────────────────────────────────────────────────────

def write_fasttext_file(
    positive: list[str],
    negative: list[str],
    path: Path,
) -> None:
    """写出 fastText 格式训练文件（每行一条样本）。"""
    rng = random.Random(42)
    lines = (
        [f"__label__wiki {t.replace(chr(10), ' ').strip()}" for t in positive if t.strip()]
        + [f"__label__cc {t.replace(chr(10), ' ').strip()}" for t in negative if t.strip()]
    )
    rng.shuffle(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("训练文件写出: %s（%d 条）", path, len(lines))


def train_classifier(train_file: Path, model_output: Path) -> None:
    """训练 fastText 二分类器并评估。"""
    logger.info("开始训练 fastText 分类器...")
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
    result = model.test(str(train_file))
    logger.info(
        "训练集评估：样本数 %d，精确率 %.4f，召回率 %.4f",
        result[0], result[1], result[2],
    )
    logger.info("模型已保存: %s", model_output)


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main() -> None:
    project_root = Path(__file__).parent.parent
    classifiers_dir = project_root / "local-shared-data" / "classifiers"
    classifiers_dir.mkdir(parents=True, exist_ok=True)
    model_path = classifiers_dir / "quality_classifier.bin"

    if model_path.exists():
        logger.info("质量分类器已存在: %s，跳过训练。", model_path)
        logger.info("如需重新训练，请先删除该文件。")
        return

    # Step 1: 采集正例
    logger.info("=" * 60)
    logger.info("Step 1/4  采集 Wikipedia 正例")
    positive_texts = fetch_wiki_articles(N_POSITIVE)

    # Step 2: 采集负例
    logger.info("=" * 60)
    logger.info("Step 2/4  采集 Common Crawl 负例")
    negative_texts = fetch_cc_negative_texts(N_NEGATIVE)

    if not positive_texts:
        logger.error("正例为空，中止训练。请检查网络连接。")
        sys.exit(1)
    if not negative_texts:
        logger.error("负例为空，中止训练。请检查 Common Crawl 下载。")
        sys.exit(1)

    # Step 3: 写训练文件 & 训练
    logger.info("=" * 60)
    logger.info("Step 3/4  写训练文件并训练分类器")
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = Path(tmpdir) / "quality_train.txt"
        write_fasttext_file(positive_texts, negative_texts, train_file)
        train_classifier(train_file, model_path)

    # Step 4: 快速验证
    logger.info("=" * 60)
    logger.info("Step 4/4  快速验证")
    import fasttext as ft  # noqa: PLC0415
    m = ft.load_model(str(model_path))

    wiki_sample = positive_texts[0][:500].replace("\n", " ")
    cc_sample = negative_texts[0][:500].replace("\n", " ")
    logger.info("维基样本预测: %s", m.predict(wiki_sample, k=1))
    logger.info("CC 样本预测:   %s", m.predict(cc_sample, k=1))

    logger.info("=" * 60)
    logger.info("完成！质量分类器已保存到: %s", model_path)


if __name__ == "__main__":
    main()
