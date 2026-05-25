# 质量分类器训练指南

本文档说明如何在**无需 Stanford 服务器账号**的情况下，在本机完整训练 Assignment 4 所需的 fastText 质量分类器，并解释每个步骤的原理。

---

## 目录

1. [为什么需要质量分类器](#1-为什么需要质量分类器)
2. [核心思路：wiki vs cc 二分类](#2-核心思路wiki-vs-cc-二分类)
3. [数据来源说明](#3-数据来源说明)
4. [运行方式](#4-运行方式)
5. [步骤详解](#5-步骤详解)
6. [训练参数说明](#6-训练参数说明)
7. [与过滤流水线集成](#7-与过滤流水线集成)
8. [替代方案：基于 LM 困惑度的质量过滤](#8-替代方案基于-lm-困惑度的质量过滤)
9. [常见问题](#9-常见问题)

---

## 1. 为什么需要质量分类器

Gopher 规则（词数、词长、省略号比例、字母占比）是**基于统计特征的硬规则**，只能过滤掉明显劣质的文本（乱码、纯数字、极短文档）。但大量低质量文本仍能通过 Gopher 过滤：

- 商品列表、导航菜单、Cookie 声明等 HTML 模板残余
- 重复的广告文案、SEO 堆砌关键词的页面
- 机器翻译或自动生成的低质量文章

质量分类器的作用是**学习"高质量文本的语言模式"**，让模型自动识别和丢弃这类低质量内容。Assignment 4 的最终目标是最小化 Paloma C4-100-domains 验证困惑度，更好的数据质量直接影响模型收敛速度和最终性能。

---

## 2. 核心思路：wiki vs cc 二分类

这一思路来自 CCNet、GPT-2 WebText、RefinedWeb 等工作：

```
正例 (__label__wiki)：高质量文本
负例 (__label__cc)  ：平均互联网质量文本
```

训练完成后，用分类器给每篇 CC 文档打分：
- 预测为 `wiki` 且置信度高 → 保留
- 预测为 `cc` → 丢弃

**为什么 Wikipedia 文章是高质量正例？**

维基百科文章由编辑社区严格审核，具有：
- 清晰的主题结构和叙述逻辑
- 有来源的事实陈述
- 无广告、无 SEO 堆砌

原版 Stanford 脚本（`train_quality_classifier.py`）使用维基百科**外链指向的页面**作为正例，逻辑更严格（被维基百科编辑引用 = 可信来源），但需要服务器上的 URL 文件和 wget 抓取。本脚本直接使用**维基百科文章本身**，效果相近。

---

## 3. 数据来源说明

### 正例：Wikipedia Featured Articles + REST API（无需账号）

**两步流程：**

**Step A**: Action API 批量获取精选条目标题
```
https://en.wikipedia.org/w/api.php
  ?action=query
  &list=categorymembers
  &cmtitle=Category:Featured articles   # 维基精选条目分类（约 6600 篇）
  &cmlimit=500                          # 每次返回 500 个标题
  &cmtype=page
  &format=json
```

**Step B**: REST API 并发获取摘要文本（10 线程并发）
```
https://en.wikipedia.org/api/rest_v1/page/summary/{title}
```

返回文章的第一段（intro paragraph），通常 200-600 字符，是维基百科最精炼的内容。

**为什么用精选条目（Featured Articles）？**
- 维基精选条目经同行评审，内容最为充实、写作质量最高
- 100% 通过 `len >= 200` 过滤（不含短存根文章）
- 约 6600 篇可用，足够采样 5000 篇正例

**为什么用 REST Summary 而非 Action API extracts？**
- Action API 的 `prop=extracts` 批量请求时，`exlimit` 被服务器强制降为 1（全文提取模式限制）
- REST API 无此限制，每篇独立请求，10 并发下达到约 10 篇/秒

### 负例：Common Crawl WET 文件（公开数据）

Common Crawl 的数据完全公开，存储在 AWS S3 和官方 HTTP 服务器上：

```
https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-17/wet.paths.gz
```

该文件列出了本次爬取（2026 年第 17 周）的所有 WET 文件路径。WET 文件是 WARC 文件的纯文本提取版本，每条记录包含一个网页的提取文本。

脚本随机选取 1 个 WET 文件（约 150MB 压缩），从中采样 5000 条文本作为负例。1 个文件包含约 10–30 万条文档，远超需求。

**Stanford 服务器上的 `english-wet-data/` 是什么？**

Stanford 的 `download_data.py` 脚本：
1. 从路径列表中随机采样 2500 个 WET 文件
2. 逐个下载、用 `is_english()` 过滤英文记录
3. 存储到 `/shared-data/english-wet-data/`

我们跳过这个步骤，直接从原始 WET 文件中采样，效果等价。

---

## 4. 运行方式

```bash
# 在 assignment4-data 目录下运行
uv run python extra_guidance/local_train_quality_classifier.py
```

**预计时间**：

| 步骤 | 时间 |
|------|------|
| Wikipedia 精选条目采集（5000篇，约 500 次 API 调用） | ~5 分钟 |
| CC WET 文件下载（~150MB，采样 10000 条） | 5–15 分钟（视网速） |
| WET 文本提取 | ~1 分钟 |
| fastText 训练 | ~30 秒 |
| **总计** | **约 15–25 分钟** |

**输出位置**：

```
local-shared-data/classifiers/quality_classifier.bin
```

**重新训练**：如需重新训练（例如调整参数），先删除现有模型文件：

```bash
rm local-shared-data/classifiers/quality_classifier.bin
uv run python extra_guidance/local_train_quality_classifier.py
```

---

## 5. 步骤详解

### Step 1/4 — Wikipedia 正例采集

```python
def fetch_wiki_articles(n: int = N_POSITIVE) -> list[str]:
```

循环调用 Wikipedia API，每次取 20 篇随机文章全文，累积到 `n` 篇为止。过滤条件：`len(text) >= 200`（过短文章可能是消歧义页面或存根）。

### Step 2/4 — Common Crawl 负例采集

```python
def fetch_cc_negative_texts(n: int = N_NEGATIVE) -> list[str]:
```

1. 下载 WET 路径列表（约 500KB），用 seed=336 随机打乱（与 Stanford 脚本一致）
2. 取第一个 WET 文件下载到临时目录（下载完成后自动删除）
3. 用 `fastwarc.ArchiveIterator` 遍历 WET 记录，筛出 `rec_type == "conversion"` 的纯文本记录
4. 过滤 `len(text) >= 200`，采满 5000 条即停止

### Step 3/4 — 写训练文件 & 训练

fastText 的训练格式：每行 `__label__xxx 文本内容`（文本必须在同一行，不能有换行符）。

```
__label__wiki Hexacyclinol is a natural metabolite of a fungus ...
__label__cc   Shop now for the best deals on electronics ...
```

训练时正负例随机打乱（seed=42），避免模型学到顺序偏差。

### Step 4/4 — 快速验证

对正负例各取一个样本运行预测，目视确认方向正确（wiki 样本预测 `__label__wiki`，cc 样本预测 `__label__cc`）。

---

## 6. 训练参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `dim` | 64 | 词向量维度。质量分类是简单二分类，64 维足够；更大维度不会显著提升精度 |
| `epoch` | 5 | 训练轮数。数据集小（6000条），5 轮已充分收敛 |
| `lr` | 0.5 | 学习率。fastText 文本分类的经验默认值 |
| `wordNgrams` | 2 | 使用 unigram + bigram 特征。Bigram 能捕捉"high quality"、"spam link"等短语模式；与 Dolma NSFW/hatespeech 模型一致 |
| `loss` | softmax | 多分类损失。对于二分类，softmax 与 sigmoid 效果接近，但 softmax 返回归一化概率更直观 |

**与 Dolma 模型的区别**：

Dolma 的 NSFW/hatespeech 模型在 Jigsaw 毒性评论数据集（Wikipedia 人工标注）上训练，样本量约 16 万，维度 100。我们的质量分类器样本量约 6000，维度 64，属于轻量版本，对于区分 wiki/cc 二分类已经足够。

---

## 7. 与过滤流水线集成

训练完成后，`cs336_data/quality.py` 中的 `classify_quality()` 函数会自动加载模型：

```python
# cs336_data/quality.py
def classify_quality(text: str) -> tuple[str, float]:
    model = _load_quality_model()   # 加载 local-shared-data/classifiers/quality_classifier.bin
    label, score = _predict(model, text)
    return (label, score)           # ("wiki", 0.85) 或 ("cc", 0.73)
```

过滤流水线 `extra_guidance/filter_pipeline.py` 中的调用：

```python
q_label, q_score = classify_quality(text)
if q_label != "wiki" or q_score < QUALITY_THRESHOLD:  # QUALITY_THRESHOLD = 0.5
    stats["dropped_quality"] += 1
    continue
```

**阈值调整**：`QUALITY_THRESHOLD` 越高，保留文本越少、质量越高，但训练数据量减少。可通过观察 Paloma 验证困惑度来寻找最优阈值。

---

## 8. 替代方案：基于 LM 困惑度的质量过滤

Assignment 4 采用 fastText 分类器，但另一种经典方案是**用训练好的语言模型计算困惑度**：

- 困惑度低 → 文本语言模式规则，符合高质量语言分布 → 保留
- 困惑度高 → 文本混乱或语言不规范 → 丢弃

**与 Assignment 1 的关联**：

Assignment 1 已训练好一个 Transformer 语言模型。如果将这个模型用于 Assignment 4 的质量过滤，流程如下：

```python
# 伪代码：基于 assignment1 模型困惑度的质量过滤
from cs336_basics.model import BasicsTransformerLM
import torch

model = BasicsTransformerLM.from_pretrained("path/to/checkpoint")
model.eval()

def perplexity_quality_filter(text: str, threshold: float = 50.0) -> bool:
    tokens = tokenizer.encode(text)[:1024]   # 取前 1024 个 token
    with torch.no_grad():
        loss = model.compute_loss(tokens)
    ppl = torch.exp(loss).item()
    return ppl < threshold  # 困惑度低于阈值则保留
```

**fastText 方案 vs 困惑度方案对比**：

| 维度 | fastText wiki/cc | LM 困惑度 |
|------|-----------------|-----------|
| 推理速度 | 极快（纯 CPU，μs 级） | 慢（需 GPU，ms 级） |
| 计算资源 | 无 GPU 需求 | 需要 GPU 或大量 CPU 时间 |
| 对 OOD 文本的鲁棒性 | 好（基于词频特征） | 差（OOD 文本困惑度虚高） |
| 训练数据依赖 | 需要 wiki/cc 标注数据 | 需要预训练好的 LM |
| Assignment 4 适用性 | 官方指定方案 ✓ | 可作为补充实验 |

Assignment 4 使用 fastText 方案是正确选择，因为需要处理数以亿计的 token，基于 LM 困惑度的过滤在实际流水线中太慢。困惑度方案更适合作为离线评估工具，用来**验证过滤后数据集整体质量的提升**，而非逐文档过滤。

---

## 9. 常见问题

**Q: WET 文件下载太慢怎么办？**

Common Crawl 数据通过 CloudFront CDN 分发，速度通常 5–20 MB/s。如果网络很慢，可以修改脚本的 `N_NEGATIVE` 从 5000 减小到 2000，只需采样更少的负例，可能 1 个 WET 文件只下载一半就够了。

**Q: Wikipedia API 被限流怎么办？**

脚本已内置 `sleep(0.5)` 限速。如果仍然遇到 429 错误，可以把 `time.sleep(0.5)` 改为 `time.sleep(2)`，速度降低但更稳定。

**Q: 质量分类器准确率怎么验证？**

最简单的验证：
```bash
uv run pytest tests/test_quality.py -v
```

更深层的验证：训练数据前后跑 Paloma 困惑度对比。如果加入质量过滤后困惑度下降，说明过滤有效。

**Q: 模型表现差（训练集精度低于 80%）怎么办？**

可能原因：正负例数量差距过大，或文本质量边界模糊。尝试：
1. 增加 `N_POSITIVE` 到 2000（更多训练轮次）
2. 增加 `FASTTEXT_EPOCHS` 到 10
3. 对 CC 负例先做 Gopher 过滤，只保留"通过 Gopher 但仍低质"的困难负例

**Q: 需要在 Stanford 服务器上重新训练吗？**

不需要。本机训练的 `quality_classifier.bin` 可以直接 `scp` 到 kong 服务器使用：
```bash
scp local-shared-data/classifiers/quality_classifier.bin \
    kong@133.9.169.119:/mnt/a/kong/workspace/ass4/local-shared-data/classifiers/
```
