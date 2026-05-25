"""
文本质量过滤模块。

包含两类质量过滤方法：

1. Gopher 质量规则（基于规则）
   来自 DeepMind Gopher 论文（Rae et al. 2021）的附录 A，
   通过文档统计量（词数、词长、省略号比例等）快速过滤明显低质量文本。

2. fastText 质量分类器（基于 ML）
   核心思路（来自 CCNet、GPT-2 WebText 等工作）：
     - 正例（wiki）：被维基百科外链引用的页面 → 通常质量可信
     - 负例（cc）：Common Crawl 随机页面 → 代表一般互联网质量
   训练脚本见 extra_guidance/train_quality_classifier.py。
"""

import functools
import re
import string
from pathlib import Path

import fasttext

from cs336_data.common import get_shared_assets_path

# ── Gopher 质量规则 ─────────────────────────────────────────────────────────────

# 词的定义（Gopher 论文 Appendix A）：
#   - 非符号词：含至少一个字母或数字字符的词
#   - 符号词：全为标点/符号
# 这里用简单的空格分割 + 字母检测来近似，比 NLTK 更快
_ALPHA_RE = re.compile(r"[a-zA-Z]")


def _word_tokenize_simple(text: str) -> list[str]:
    """简单空格分词，不引入 NLTK 依赖，速度快。"""
    return text.split()


def gopher_quality_filter(text: str) -> bool:
    """判断文本是否通过 Gopher 质量过滤器。

    实现 Gopher 论文中的以下规则（满足任一条件则丢弃）：
      1. 非符号词数量 < 50 或 > 100,000
      2. 平均非符号词长度 不在 [3, 10] 范围内
      3. 以省略号 "..." 结尾的行比例 > 30%
      4. 含至少一个字母字符的词占比 < 80%

    Args:
        text: 待过滤的纯文本字符串。

    Returns:
        True 表示通过（保留），False 表示不通过（丢弃）。
    """
    words = _word_tokenize_simple(text)
    if not words:
        return False

    # ── 规则 1：词数范围 ───────────────────────────────────────────────────────
    # 非符号词 = 含至少一个字母的词
    non_symbol_words = [w for w in words if _ALPHA_RE.search(w)]
    n_non_symbol = len(non_symbol_words)
    if n_non_symbol < 50 or n_non_symbol > 100_000:
        return False

    # ── 规则 2：平均词长 ──────────────────────────────────────────────────────
    avg_word_len = sum(len(w) for w in non_symbol_words) / n_non_symbol
    if avg_word_len < 3 or avg_word_len > 10:
        return False

    # ── 规则 3：省略号行占比 ─────────────────────────────────────────────────
    lines = text.splitlines()
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.3:
            return False

    # ── 规则 4：含字母词占比 ─────────────────────────────────────────────────
    alpha_words = sum(1 for w in words if _ALPHA_RE.search(w))
    if alpha_words / len(words) < 0.8:
        return False

    return True


# ── fastText 质量分类器 ─────────────────────────────────────────────────────────

_QUALITY_MODEL_PATH = "classifiers/quality_classifier.bin"
_LABEL_PREFIX = "__label__"


@functools.lru_cache(maxsize=1)
def _load_quality_model() -> fasttext.FastText._FastText:
    """懒加载质量分类器（需先运行 extra_guidance/train_quality_classifier.py 训练）。"""
    model_path = get_shared_assets_path() / _QUALITY_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"质量分类器不存在: {model_path}\n"
            "请先运行训练脚本：\n"
            "  uv run python extra_guidance/train_quality_classifier.py\n"
            "详细说明见 extra_guidance/CODE_WALKTHROUGH.md"
        )
    return fasttext.load_model(str(model_path))


def classify_quality(text: str) -> tuple[str, float]:
    """判断文本的质量（高质量 wiki vs 低质量 cc）。

    使用预训练的 fastText 分类器，标签为：
      - "wiki"：高质量（正例，维基百科引用的页面）
      - "cc"  ：低质量（负例，随机 Common Crawl 页面）

    Args:
        text: 待分类的 Unicode 字符串（页面全文）。

    Returns:
        ("wiki", score) 或 ("cc", score)，score 为 0~1 置信度。
    """
    model = _load_quality_model()
    # Truncate to first 2000 chars: model was trained on Wikipedia summaries
    # (~200-600 chars); very long texts dilute the wiki-style signal.
    clean_text = text.replace("\n", " ").strip()[:2000]
    if not clean_text:
        return ("cc", 1.0)
    labels, probs = model.predict(clean_text, k=1)
    label = labels[0].replace(_LABEL_PREFIX, "")
    return (label, float(probs[0]))
