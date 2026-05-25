"""
有害内容分类模块（NSFW + 有毒言论）。

使用 Dolma 项目发布的 fastText 二分类器，在 Jigsaw 毒性评论数据集
（Wikipedia 评论，含多种有害标签）上训练。推理速度极快（纯 CPU）。

模型位置（通过 get_shared_assets_path() 获取本地或远程路径）：
  - NSFW:       classifiers/dolma_fasttext_nsfw_jigsaw_model.bin
  - 仇恨言论:   classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin

原始模型下载地址（Dolma artifacts）：
  - dolma-artifacts.org/.../jigsaw_fasttext_bigrams_nsfw_final.bin
  - dolma-artifacts.org/.../jigsaw_fasttext_bigrams_hatespeech_final.bin
"""

import functools

import fasttext

from cs336_data.common import get_shared_assets_path

_LABEL_PREFIX = "__label__"


def _load_model(model_filename: str) -> fasttext.FastText._FastText:
    """加载指定的 fastText 分类模型（带路径校验和友好错误提示）。"""
    model_path = get_shared_assets_path() / "classifiers" / model_filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"分类模型不存在: {model_path}\n"
            "请将模型文件放置在 local-shared-data/classifiers/ 目录下。\n"
            "详见 extra_guidance/REPORT.md 中的数据准备说明。"
        )
    return fasttext.load_model(str(model_path))


@functools.lru_cache(maxsize=1)
def _load_nsfw_model() -> fasttext.FastText._FastText:
    return _load_model("dolma_fasttext_nsfw_jigsaw_model.bin")


@functools.lru_cache(maxsize=1)
def _load_hatespeech_model() -> fasttext.FastText._FastText:
    return _load_model("dolma_fasttext_hatespeech_jigsaw_model.bin")


def _predict(model: fasttext.FastText._FastText, text: str) -> tuple[str, float]:
    """通用预测函数：返回 top-1 标签（去掉 __label__ 前缀）和置信度。"""
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return ("unknown", 0.0)
    labels, probs = model.predict(clean_text, k=1)
    label = labels[0].replace(_LABEL_PREFIX, "")
    return (label, float(probs[0]))


def classify_nsfw(text: str) -> tuple[str, float]:
    """判断文本是否包含 NSFW（不适合工作场所）内容。

    NSFW 定义：色情、亵渎或其他可能令人不安的内容。

    Args:
        text: 待分类的 Unicode 字符串（页面全文）。

    Returns:
        ("nsfw", score) 或 ("non-nsfw", score)，score 为 0~1 置信度。
    """
    model = _load_nsfw_model()
    return _predict(model, text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """判断文本是否包含有毒言论。

    有毒言论定义：粗鲁、不尊重或不合理的、可能使人离开讨论的语言。

    Args:
        text: 待分类的 Unicode 字符串（页面全文）。

    Returns:
        ("toxic", score) 或 ("non-toxic", score)，score 为 0~1 置信度。
    """
    model = _load_hatespeech_model()
    return _predict(model, text)
