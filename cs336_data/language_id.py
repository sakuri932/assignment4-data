"""
语言识别模块。

使用 fastText 的 lid.176.bin 模型，支持 176 种语言的识别。
该模型由 Facebook AI 团队训练，在 Common Crawl 过滤场景中被
Dolma、RefinedWeb、LLaMA 等主流数据集广泛采用。

语言过滤用途：
  - 将置信度 >= 0.7 的英文页面判定为英文（与 CC 官方过滤标准一致）
  - 分类器给出 0~1 的置信度分数，可作为过滤阈值

模型路径优先级：
  1. /shared-data/classifiers/lid.176.bin  （远程 Modal 环境）
  2. local-shared-data/classifiers/lid.176.bin  （本地开发环境）
"""

import functools
import re
from pathlib import Path

import fasttext

from cs336_data.common import get_shared_assets_path

# fastText 返回的标签格式为 "__label__zh"，需要去掉前缀
_LABEL_PREFIX = "__label__"

# 中文繁简体的 fastText 标签映射到统一的 "zh"
_LANG_REMAP: dict[str, str] = {
    "zh": "zh",
    "zht": "zh",       # 繁体中文（部分模型版本）
    "zh-hans": "zh",
    "zh-hant": "zh",
}


@functools.lru_cache(maxsize=1)
def _load_model() -> fasttext.FastText._FastText:
    """懒加载语言识别模型（全局只加载一次，后续调用直接返回缓存）。这里加载的是 fastText 的 lid.176.bin 模型，支持 176 种语言的识别。"""
    model_path = get_shared_assets_path() / "classifiers" / "lid.176.bin"
    if not model_path.exists():
        raise FileNotFoundError(
            f"语言识别模型不存在: {model_path}\n"
            "请运行: wget -O {model_path} "
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        )
    # suppress_output=True：屏蔽 fastText 加载时的警告信息
    return fasttext.load_model(str(model_path))


def identify_language(text: str) -> tuple[str, float]:
    """识别文本的主要语言。

    Args:
        text: 待识别的 Unicode 字符串。

    Returns:
        (language_code, confidence) 元组，如 ("en", 0.98)。
        language_code 为 ISO 639-1 代码，已做中文繁简体统一映射。
        confidence 为 0~1 之间的置信度（fastText 原始 softmax 概率）。
    """
    model = _load_model()

    # fastText 不接受换行符，需替换为空格
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return ("unknown", 0.0)

    # predict 返回 ([labels], [probs])，k=1 只取 top-1
    labels, probs = model.predict(clean_text, k=1)

    lang = labels[0].replace(_LABEL_PREFIX, "")
    score = float(probs[0])

    # 中文繁简体统一为 "zh"
    lang = _LANG_REMAP.get(lang, lang)

    return (lang, score)


def is_english(text: str, threshold: float = 0.7) -> bool:
    """判断文本是否为英文（置信度 >= threshold）。

    用于 EnglishWetFile._create 中对 WET 记录的语言过滤。
    """
    lang, score = identify_language(text)
    return lang == "en" and score >= threshold
