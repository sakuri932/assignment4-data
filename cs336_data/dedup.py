"""
文档去重模块。

包含两种去重方法：

1. 精确行去重（Exact Line Deduplication）
   对跨文件重复的行计数，删除出现超过 1 次的行。
   适合去除导航栏、页脚、免责声明等模板化内容。

2. MinHash + LSH 模糊文档去重（Fuzzy Document Deduplication）
   基于 Jaccard 相似度的近似去重，能处理"内容几乎相同但有细微差异"的重复文档。
   算法步骤：
     a. 文本归一化（小写/去标点/去变音符/NFD）
     b. 生成词级 n-gram 集合
     c. 计算 MinHash 签名（k 个哈希函数，每个取最小哈希值）
     d. LSH 分桶（将签名分为 b 个条带，每个条带单独哈希）
     e. 候选对筛选（同桶文档 = 候选重复）
     f. 真实 Jaccard 相似度验证
     g. Union-Find 聚类 + 随机保留一篇
"""

import os
import random
import string
import unicodedata
from collections import defaultdict
from pathlib import Path

import mmh3  # MurmurHash3，固定大小哈希，适合大规模计数

# ── 精确行去重 ────────────────────────────────────────────────────────────────


def exact_line_deduplication(
    input_files: list[os.PathLike],
    output_directory: os.PathLike,
) -> None:
    """跨文件精确行去重。

    算法：
      Pass 1：统计所有文件中每行的全局出现次数（使用 MurmurHash3 减少内存）
      Pass 2：重写每个文件，只保留全局出现次数恰好为 1 的行

    Args:
        input_files: 输入文件路径列表。
        output_directory: 输出目录（输出文件与输入文件同名）。
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1：统计每行哈希的全局出现次数
    line_counts: dict[int, int] = defaultdict(int)
    for path in input_files:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                # 使用 128-bit MurmurHash3 降低碰撞概率
                h = mmh3.hash128(line, signed=False)
                line_counts[h] += 1

    # Pass 2：重写文件，只保留全局唯一行
    for path in input_files:
        out_path = output_dir / Path(path).name
        with open(path, encoding="utf-8", errors="replace") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                h = mmh3.hash128(line, signed=False)
                if line_counts[h] == 1:
                    fout.write(line)


# ── MinHash + LSH 模糊文档去重 ────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    """文本归一化（参照 RefinedWeb/Falcon 论文 [7] 的建议）。

    步骤：
      1. NFD Unicode 规范化（将合成字符分解为基础字符 + 变音符组合）
      2. 去掉变音符（Unicode 类别 Mn = 非间距标记）
      3. 转小写
      4. 去掉标点
      5. 规范化空白（多个空格合并为一个）
    """
    # NFD 分解后去掉变音符（如 é → e + ́ → e）
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def _get_word_ngrams(text: str, n: int) -> list[str]:
    """将文本切分为词级 n-gram 列表（允许重复，因为要计算 Jaccard）。"""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _compute_minhash_signature(ngram_set: set[str], num_hashes: int) -> list[int]:
    """计算 n-gram 集合的 MinHash 签名。

    使用 num_hashes 个独立哈希函数（不同 seed 的 MurmurHash3），
    每个函数对 n-gram 集合取最小哈希值。

    数学性质：
      P[minhash_i(A) == minhash_i(B)] = Jaccard(A, B)
    即两文档签名在某位置相等的概率等于其 Jaccard 相似度。

    Args:
        ngram_set: 文档 n-gram 集合。
        num_hashes: 哈希函数数量（即签名维度 k）。

    Returns:
        长度为 num_hashes 的整数列表（签名向量）。
    """
    signature = []
    ngrams_list = list(ngram_set)
    for seed in range(num_hashes):
        # 取所有 n-gram 在当前哈希函数下的最小值
        min_hash = min(
            (mmh3.hash(ngram, seed=seed, signed=False) for ngram in ngrams_list),
            default=0,
        )
        signature.append(min_hash)
    return signature


def _lsh_bucket_keys(signature: list[int], num_bands: int) -> list[tuple]:
    """将签名按 LSH 分带策略转换为桶键。

    将 k 维签名分为 b = num_bands 个条带，每个条带含 r = k/b 个值。
    每个条带的桶键为 (band_idx, hash(条带值))。

    两文档若在任意一个条带内完全一致，则进入同一桶（候选重复对）。

    Args:
        signature: MinHash 签名（长度 k = num_bands * r）。
        num_bands: 分带数 b。

    Returns:
        长度为 num_bands 的桶键列表。
    """
    k = len(signature)
    r = k // num_bands
    keys = []
    for band_idx in range(num_bands):
        band = tuple(signature[band_idx * r : (band_idx + 1) * r])
        # 将 (条带索引, 条带值) 作为桶键，避免不同位置的相同条带误合并
        keys.append((band_idx, band))
    return keys


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """计算两个集合的真实 Jaccard 相似度。"""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class _UnionFind:
    """路径压缩 + 按秩合并的 Union-Find 结构，用于聚类重复文档。"""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # 按秩合并：小树挂到大树上
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    """使用 MinHash + LSH 对文档集合进行模糊去重。

    每个输入文件视为一个"文档"（整个文件内容 = 一篇文档）。

    算法流程：
      1. 读取每个文件，归一化文本
      2. 生成词级 n-gram 集合
      3. 计算 MinHash 签名（k = num_hashes 个哈希函数）
      4. LSH 分桶：b = num_bands 个条带，r = k/b 个哈希/条带
      5. 同桶文档为候选重复对
      6. 计算候选对的真实 Jaccard 相似度
      7. 相似度 > jaccard_threshold 的对 → Union-Find 合并为同一簇
      8. 每个簇随机保留一个文档，其余丢弃
      9. 将保留文档写入输出目录

    Args:
        input_files: 输入文件路径列表。
        num_hashes: MinHash 签名维度 k（必须能被 num_bands 整除）。
        num_bands: LSH 分带数 b。
        ngrams: n-gram 的 n（词数）。
        jaccard_threshold: 真实 Jaccard 相似度阈值，超过则视为重复。
        output_directory: 输出目录。
    """
    assert num_hashes % num_bands == 0, "num_hashes 必须能被 num_bands 整除"

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1：读取文件，计算签名 ─────────────────────────────────────────
    file_paths = [Path(p) for p in input_files]
    doc_texts: list[str] = []
    doc_ngram_sets: list[set[str]] = []
    doc_signatures: list[list[int]] = []

    for path in file_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        norm_text = _normalize_text(text)
        ngram_list = _get_word_ngrams(norm_text, ngrams)
        ngram_set = set(ngram_list)
        signature = _compute_minhash_signature(ngram_set, num_hashes)

        doc_texts.append(text)
        doc_ngram_sets.append(ngram_set)
        doc_signatures.append(signature)

    # ── Step 2：LSH 分桶，收集候选重复对 ───────────────────────────────────
    # buckets: {桶键 → 文档索引列表}
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for doc_idx, signature in enumerate(doc_signatures):
        for key in _lsh_bucket_keys(signature, num_bands):
            buckets[key].append(doc_idx)

    # ── Step 3：验证候选对，找出真实重复 ────────────────────────────────────
    uf = _UnionFind()
    seen_pairs: set[tuple[int, int]] = set()  # 防止重复计算同一对

    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                pair = (min(a, b), max(a, b))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # 计算真实 Jaccard 相似度
                sim = _jaccard_similarity(doc_ngram_sets[a], doc_ngram_sets[b])
                if sim >= jaccard_threshold:
                    uf.union(a, b)

    # ── Step 4：确定每个簇的保留文档 ────────────────────────────────────────
    # 将文档按簇根分组
    clusters: dict[int, list[int]] = defaultdict(list)
    for doc_idx in range(len(file_paths)):
        root = uf.find(doc_idx)
        clusters[root].append(doc_idx)

    # 每个簇随机保留一个（固定种子保证可重复性）
    rng = random.Random(42)
    kept: set[int] = set()
    for cluster in clusters.values():
        kept.add(rng.choice(cluster))

    # ── Step 5：写出保留的文档 ──────────────────────────────────────────────
    for doc_idx, path in enumerate(file_paths):
        if doc_idx in kept:
            out_path = output_dir / path.name
            out_path.write_text(doc_texts[doc_idx], encoding="utf-8")
