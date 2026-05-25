# CS336 Assignment 4 代码解读

> 本文档逐模块、逐函数详解 Assignment 4 的所有代码实现，包括设计决策、算法原理和关键细节。

---

## 目录

1. [cs336_data/extract.py — HTML 转文本](#1-cs336_dataextractpy--html-转文本)
2. [cs336_data/language_id.py — 语言识别](#2-cs336_datalanguage_idpy--语言识别)
3. [cs336_data/pii.py — PII 屏蔽](#3-cs336_datapipy--pii-屏蔽)
4. [cs336_data/harmful.py — 有害内容分类](#4-cs336_dataharmfulpy--有害内容分类)
5. [cs336_data/quality.py — 质量过滤](#5-cs336_dataqualitypy--质量过滤)
6. [cs336_data/dedup.py — 去重](#6-cs336_datadeduppy--去重)
7. [tests/adapters.py — 测试适配器](#7-testsadapterspy--测试适配器)
8. [extra_guidance/train_quality_classifier.py — 质量分类器训练](#8-extra_guidancetrain_quality_classifierpy--质量分类器训练)
9. [extra_guidance/filter_pipeline.py — 端到端过滤流水线](#9-extra_guidancefilter_pipelinepy--端到端过滤流水线)
10. [extra_guidance/tokenize_data.py — 分词序列化](#10-extra_guidancetokenize_datapy--分词序列化)

---

## 1 `cs336_data/extract.py` — HTML 转文本

### 为什么需要 HTML 提取？

Common Crawl 的 WARC 文件存储的是浏览器拿到的原始 HTML，其中充斥着 `<script>`、`<style>`、`<nav>`、`<footer>` 等标签，这些标签内的文本对语言模型来说没有语义价值，甚至有害（如 JavaScript 代码）。提取主体文本是数据清洗的第一步。

### 库选择：Resiliparse

- **速度**：底层 C++ 实现，比 BeautifulSoup（纯 Python）快约 10 倍
- **容错性**：网络上大量 HTML 不符合规范（未闭合标签、错误嵌套），Resiliparse 用 libxml2 / lxml 的容错解析器处理
- **编码检测**：附带 `detect_encoding` 函数，能识别 Latin-1、GBK、Shift-JIS 等非 UTF-8 编码

### 关键函数：`extract_text_from_html_bytes`

```python
def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding)
    return extract_plain_text(html_str)
```

**为什么先试 UTF-8 再检测？**  
UTF-8 覆盖约 98.2% 的网页（Wikipedia 数据），每次都调用 `detect_encoding` 反而更慢。用 try/except 做快速路径是标准做法。

**`extract_plain_text` 的行为：**  
默认参数下，它会：
- 保留 `<h1>`-`<h6>`、`<p>`、`<li>` 等内容标签中的文本
- 丢弃 `<script>`、`<style>`、`<noscript>` 等非内容标签
- 保留列表项的缩进格式（如 `  • Novel`）
- 将 HTML 实体（如 `&amp;`）解码回正常字符

---

## 2 `cs336_data/language_id.py` — 语言识别

### 背景

网络内容覆盖 7000+ 种语言，但当前计算预算下的语言模型通常只能专注几种语言。识别语言并过滤是构建单语言训练集的必要步骤。

### fastText lid.176.bin 模型

- 由 Facebook AI 团队训练，支持 176 种语言
- 基于字符级 n-gram + 词级 n-gram 的浅层线性分类器
- 推理速度极快（亚毫秒级），适合大规模过滤
- Dolma、RefinedWeb、LLaMA 训练数据均使用此模型

### 关键实现细节

#### `lru_cache` 懒加载

```python
@functools.lru_cache(maxsize=1)
def _load_model() -> fasttext.FastText._FastText:
    ...
```

模型文件约 128MB，只加载一次，后续调用直接返回缓存对象。在过滤流水线中，每个进程会各自加载一次（`lru_cache` 是进程级别的）。

#### 中文统一映射

```python
_LANG_REMAP = {"zh": "zh", "zht": "zh", "zh-hans": "zh", "zh-hant": "zh"}
```

fastText 模型可能返回 `zh`（简体中文）或 `zh-hant`（繁体中文）等变体，统一映射为 `zh` 以匹配测试用例。

#### 换行符处理

```python
clean_text = text.replace("\n", " ").strip()
```

fastText 以换行符为样本分隔符，输入文本不能包含换行，否则只识别第一行。

#### 过滤阈值

`is_english(text, threshold=0.7)` 使用 0.7 的置信度阈值，与 CC 官方 WET 预过滤标准一致（PDF Section 4 中提到课程组以此标准预处理了 2500 个 WET 文件）。

---

## 3 `cs336_data/pii.py` — PII 屏蔽

### 为什么需要 PII 屏蔽？

语言模型在训练时记忆输入数据，在生成时可能复现真实用户的邮件地址、电话等私人信息，造成隐私泄露。屏蔽而非删除（保留占位符）的策略优于删除：
- **保留句子完整性**：模型不会在原位置遇到奇怪的语法断裂
- **隐式学习**：模型学到"这里曾有一个邮件地址"的位置信息
- **可逆性**：原始数据中的占位符可以在需要时被替换

### 电子邮件正则

```python
_EMAIL_RE = re.compile(
    r"(?<!\|\|\|)"
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+"
    r"\.[a-zA-Z]{2,6}"
)
```

- 本地部分允许：字母、数字、`.`、`_`、`%`、`+`、`-`
- 负向后顾 `(?<!\|\|\|)` 防止匹配已有占位符中的内容（虽然占位符中无 `@`，但作为防御性编程）
- TLD 限制 2-6 字符（`.com`/`.museum` 等）

### 电话号码正则

```python
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\d{10}|(?:\+?1[\s.\-]?)?(?:\(\d{3}\)[\s.\-]?|\d{3}[\s.\-])\d{3}[\s.\-]?\d{4})"
    r"(?!\d)"
)
```

**两个分支的必要性：**
- 分支 1 `\d{10}`：捕获如 `2831823829` 的纯数字格式
- 分支 2：捕获带括号/连字符/空格的格式

**边界保护：**
- `(?<!\d)` 和 `(?!\d)` 防止从较长数字串（如信用卡号、ZIP+4 邮编）中错误截取

### IPv4 正则

```python
_OCTET = r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
_IPV4_RE = re.compile(rf"(?<!\d)(?:{_OCTET}\.){{3}}{_OCTET}(?!\d)")
```

精确匹配 0-255 范围的 4 段数字，避免匹配如 `192.0.2.1000` 这样的无效 IP。每段匹配：
- `25[0-5]`：250-255
- `2[0-4]\d`：200-249
- `[01]?\d\d?`：0-199

---

## 4 `cs336_data/harmful.py` — 有害内容分类

### 分类器来源

来自 Dolma 项目（AI2），在 Jigsaw Toxic Comments 数据集上用 fastText 训练：
- **Jigsaw 数据集**：约 16 万条 Wikipedia 编辑评论，人工标注毒性/色情/仇恨等多个维度
- **fastText bigram 模型**：比深度模型推理速度快 100 倍以上，但对于这种有明显词汇特征的分类任务，效果接近

### 两个模型的区别

- `dolma_fasttext_nsfw_jigsaw_model.bin`：检测色情/亵渎内容（trained on `obscene` Jigsaw label）
- `dolma_fasttext_hatespeech_jigsaw_model.bin`：检测仇恨言论/有毒内容（trained on `toxic` Jigsaw label）

### 局限性（PDF Section 2.5 问题 c）

1. **分布偏移**：Jigsaw 数据来自 Wikipedia 评论，与一般网页文本风格差异大，可能误杀/漏杀
2. **上下文敏感**：医学、历史、法律文本中的描述可能被误判
3. **多语言失效**：模型主要在英文上训练，非英文有害内容可能漏判

---

## 5 `cs336_data/quality.py` — 质量过滤

### 5.1 Gopher 质量规则

来自 DeepMind 的 Gopher 论文（Rae et al. 2021）附录 A，以下四条规则实现完整：

| 规则 | 阈值 | 原理 |
|------|------|------|
| 非符号词数量 | [50, 100,000] | 过短 = 碎片页面；过长 = 可能是脚本/日志 |
| 平均词长 | [3, 10] 字符 | 平均词长 < 3 = 乱码/代码；> 10 = 技术文本/乱码 |
| 省略号行比例 | < 30% | 大量省略号 = 目录页/付费墙截断页 |
| 含字母词比例 | ≥ 80% | 比例低 = 大量数字/符号，非自然语言文本 |

**非符号词的定义：** 含至少一个字母字符（`re.search(r"[a-zA-Z]", word)`）的词。这排除纯数字、纯标点等。

**为什么这些规则有效？**

它们捕捉了几类典型的低质量网页：
- **"Our servers have detected..."**：词数 < 50 → 被规则 1 过滤
- **导航菜单**：大量短词（"Home", "Menu"）或纯符号 → 规则 2/4
- **分页预告**："More content..." × 50 行 → 规则 3

### 5.2 质量分类器

**核心洞察（来自 CCNet、GPT-2 WebText、LLaMA）：**

链接信号是内容质量的代理变量：
- 被维基百科外链引用的页面 → 经过了人工编辑审核 → 通常信息密度高、格式规范
- 随机 Common Crawl 页面 → 未经筛选 → 质量参差不齐

**训练流程（详见 `train_quality_classifier.py`）：**

```
维基百科外链 URL → wget 抓取 WARC → Resiliparse 提取文本 → __label__wiki
CC WET 文件       → 随机采样              → 直接读取文本   → __label__cc
                   ↓
              fastText 训练（bigram, dim=64, epoch=5）
                   ↓
         local-shared-data/classifiers/quality_classifier.bin
```

**设置阈值：**

训练完成后，可通过 Paloma 验证集调整分类阈值（如只保留 "wiki" 预测且置信度 > 0.7 的页面），在过滤比例和数据质量之间权衡。

---

## 6 `cs336_data/dedup.py` — 去重

### 6.1 精确行去重

**场景：** 去除跨文档重复的模板性内容（页眉、导航、版权声明等）

**算法：**

```
Pass 1：对每行内容计算 MurmurHash3-128，统计全局计数
Pass 2：重写每个文件，只保留计数 == 1 的行
```

**为什么用哈希而不是直接存字符串？**

MurmurHash3-128 将任意长度字符串映射为 16 字节固定大小整数，内存效率提升约 10 倍。以 100 万行、平均 50 字节/行为例：
- 直接存字符串：~50MB
- 存 128-bit 哈希：~16MB

使用 `mmh3.hash128(line, signed=False)` 产生 128 位无符号整数，碰撞概率约 $2^{-128}$，可以忽略。

**关键细节：**

读文件时保留行末换行符（`for line in f` 默认保留 `\n`），写出时原样保留。这保证了输出文件的结构与输入一致（不会人为添加/删除换行）。

### 6.2 MinHash + LSH 文档去重

**为什么精确去重不够？**

MIT 许可证文件：每个项目只有年份和作者名不同，但内容 95% 相同。精确行匹配无法识别这类"模板相同，填充不同"的重复。

#### MinHash 算法

**Jaccard 相似度：**

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

对两个文档的 n-gram 集合，Jaccard 相似度衡量它们内容的重叠程度。

**MinHash 的核心性质：**

对于哈希函数 $h$，从集合 $A$ 中取哈希值最小的元素：

$$\text{minhash}(h, A) = \min_{a \in A} h(a)$$

可以证明：

$$P[\text{minhash}(h, A) = \text{minhash}(h, B)] = J(A, B)$$

即两文档 minhash 值相等的概率等于它们的 Jaccard 相似度。用 $k$ 个独立哈希函数构成签名向量，签名中相同元素的比例就是 Jaccard 的无偏估计。

**实现：**

```python
for seed in range(num_hashes):
    min_hash = min(
        mmh3.hash(ngram, seed=seed, signed=False)
        for ngram in ngram_set
    )
    signature.append(min_hash)
```

每个 seed 对应一个独立哈希函数（MurmurHash3 对不同 seed 产生独立、均匀的哈希）。

#### LSH 算法

**问题：** 有 $n$ 个文档，暴力比较所有对是 $O(n^2)$，对百亿文档不可行。

**LSH 思路：** 将 $k$ 维签名分成 $b$ 个条带，每个条带有 $r = k/b$ 个值。两文档若在任意一个条带内完全相同，则进入同一"桶"，成为候选对。

**为什么这样有效？**

对于 Jaccard 相似度为 $s$ 的两个文档：
- 在某个哈希函数下 minhash 相同的概率 = $s$
- 在某个条带（$r$ 个哈希）下完全相同的概率 = $s^r$
- 在所有 $b$ 个条带下都不相同的概率 = $(1 - s^r)^b$
- 进入至少一个桶（被识别为候选对）的概率 = $1 - (1 - s^r)^b$

通过调整 $b$ 和 $r$，可以控制精确率-召回率的平衡：
- 增大 $b$（更多条带）→ 更多候选对 → 召回率高、精确率低
- 增大 $r$（每带更多哈希）→ 更少候选对 → 精确率高、召回率低

#### Union-Find 聚类

```python
class _UnionFind:
    def find(self, x):  # 路径压缩
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):  # 按秩合并
        px, py = self.find(x), self.find(y)
        if rank[px] < rank[py]: px, py = py, px
        self.parent[py] = px
```

路径压缩 + 按秩合并使 `find` 和 `union` 的摊还时间复杂度接近 $O(1)$。

#### 文本归一化

在计算 MinHash 之前对文本归一化（参照 RefinedWeb/Falcon 论文）：
1. NFD Unicode 规范化：将 `é` 分解为 `e` + 组合重音符
2. 去掉变音符（Unicode 类别 Mn）：`é` → `e`
3. 转小写
4. 去掉标点
5. 规范化空白

这确保了只有空格、大小写、标点差异的文档能被正确识别为重复。

---

## 7 `tests/adapters.py` — 测试适配器

适配器层将实现代码与测试框架解耦：

- 测试文件只 `import` 适配器函数（稳定接口）
- 实现可以随时更改内部逻辑，不破坏测试签名
- 适配器可以在此处做语言代码重映射、格式转换等轻量处理

所有适配器都用延迟导入（在函数体内 `from ... import`），确保：
1. 测试文件加载时不会因为缺失模型文件而崩溃
2. 只有实际运行的测试才会触发模型加载

---

## 8 `extra_guidance/train_quality_classifier.py` — 质量分类器训练

### 数据采集策略

**正例采集：**
1. 从 `enwiki-20260501-extracted_urls.txt.gz` 读取维基百科外链 URL
2. 随机采样 10,000 个 URL
3. 用 `wget` 抓取为 WARC 格式（`--timeout=5` 避免慢站拖延）
4. 用 Resiliparse 从 WARC 中提取文本
5. 对正例也应用 Gopher 过滤（去掉损坏链接的 404 页面等）

**负例采集：**
从 `english-wet-data/` 中随机读取 10,000 篇文档，直接作为负例。

**为什么正负例数量相等？**

fastText 对类别不平衡敏感。相等数量使分类器不偏向任何一类。

### fastText 超参数

```python
fasttext.train_supervised(
    dim=64,              # 词向量维度（64 足够区分两类）
    epoch=5,             # 通常 5 轮足够收敛
    lr=0.5,              # 标准初始学习率
    wordNgrams=2,        # bigram，捕捉短语特征（"machine learning"等）
    loss="softmax",      # 多分类损失（这里是二分类）
)
```

---

## 9 `extra_guidance/filter_pipeline.py` — 端到端过滤流水线

### 过滤步骤顺序设计

过滤步骤的顺序影响效率：**把最快、丢弃最多的过滤器放前面**

```
WET 记录 (100%)
  → Gopher 质量过滤（~30-50% 丢弃，纯 Python，最快）
  → 语言确认（~5% 丢弃，fastText，极快）
  → NSFW 过滤（~2-5% 丢弃，fastText）
  → 仇恨言论过滤（~1-3% 丢弃，fastText）
  → 质量分类器（~50-70% 丢弃，fastText）
  → PII 屏蔽（不丢弃，仅替换，放最后避免浪费）
  → 保留 (~10-20%)
```

### 并行策略

使用 `concurrent.futures.ProcessPoolExecutor`，每个进程处理一个 WET 文件。

**为什么用进程而不是线程？**

Python GIL（全局解释器锁）阻止多线程真正并行执行 CPU 密集型代码（如 fastText 推理）。多进程绕过 GIL，每个进程有独立内存空间和 GIL。

**子进程的模型加载：**

每个子进程在第一次调用时触发 `lru_cache` 初始化，加载模型。128MB 的模型 × 8 个进程 = ~1GB 内存，在有 16GB+ 内存的服务器上完全可接受。

---

## 10 `extra_guidance/tokenize_data.py` — 分词序列化

### 为什么用 GPT-2 分词器？

训练脚本 `scripts/train.py` 使用 GPT-2 tokenizer（词表 50,257 个 token），Paloma 验证数据也用相同分词器编码（`tokenized_paloma_c4_100_domains_validation.bin`），保持一致性。

### uint16 格式

GPT-2 词表 50,257 < 65,536 = $2^{16}$，uint16（2 字节/token）足够存储所有 token ID。相比 int32（4 字节/token）节省一半存储空间和 I/O 带宽。

### EOS Token

```python
return tokenizer.encode(text) + [tokenizer.eos_token_id]
```

GPT-2 的 `<|endoftext|>` = token ID 50256。在每篇文档末尾加 EOS 有两个作用：
1. 告诉模型文档在这里结束，不要跨文档生成
2. 帮助模型学习何时停止生成（inference 时用于检测终止）

### 文档分割

过滤脚本以双空行（`\n\n`）作为文档边界，分词脚本用 `text.split("\n\n")` 切分，对每篇文档分别 tokenize 后拼接。
