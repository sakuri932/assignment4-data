# CS336 Assignment 4（数据）：过滤语言模型训练数据

**版本 26.0.1 | CS336 课程团队 | Spring 2026**

---

## 目录

1. [作业概述](#1-作业概述)
2. [过滤 Common Crawl](#2-过滤-common-crawl)
   - [2.1 观察数据](#21-观察数据)
   - [2.2 HTML 转文本](#22-html-转文本)
   - [2.3 语言识别](#23-语言识别)
   - [2.4 个人可识别信息](#24-个人可识别信息)
   - [2.5 有害内容](#25-有害内容)
   - [2.6 质量规则](#26-质量规则)
   - [2.7 质量分类器](#27-质量分类器)
3. [去重](#3-去重)
   - [3.1 精确行去重](#31-精确行去重)
   - [3.2 MinHash + LSH 文档去重](#32-minhash--lsh-文档去重)
4. [过滤语言模型训练数据](#4-过滤语言模型训练数据)
5. [参考文献](#5-参考文献)

---

## 1 作业概述

在本作业中，你将获得对网页爬取数据进行过滤、为语言模型准备训练数据的实践经验。

**你将实现以下内容：**

1. 将 Common Crawl HTML 转换为纯文本。
2. 用多种方法过滤提取的文本（如有害内容、个人可识别信息等）。
3. 对训练数据去重。

**你将运行以下内容：**

1. 在不同数据集上训练语言模型，以更好地理解特定处理决策对模型性能的影响。

**代码结构如下：**

所有作业代码及本说明文档均可在 GitHub 获取：

[github.com/stanford-cs336/assignment4-data](https://github.com/stanford-cs336/assignment4-data)

请 `git clone` 该仓库；如有更新，课程组会通知你，你可以 `git pull` 获取最新版本。

1. `cs336_basics/*`：包含作业 1 中构建的模型训练代码，略有优化——将部分手工实现的组件替换为 PyTorch 原生等价实现（如使用 PyTorch 内置的交叉熵核）。还包含支持多 GPU 分布式数据并行训练的训练脚本，你将用它在过滤后的数据上训练模型。

2. `cs336_data/*`：这是你编写作业 4 代码的地方。课程组创建了名为 `cs336_data` 的模块，其中包含起始代码和占位符。

3. `tests/*.py`：包含你必须通过的所有测试。这些测试调用 `tests/adapters.py` 中定义的钩子，你需要实现适配器来将你的代码连接到测试。可以额外编写或修改测试代码来辅助调试，但你的实现必须通过原始提供的测试套件。

4. `README.md`：包含关于目录结构的更多细节，以及配置环境的基本说明。

**如何提交：**

你需要向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有书面问题，请使用排版软件。
- `code.zip`：包含你编写的所有代码。

---

## 2 过滤 Common Crawl

大型语言模型主要在互联网数据上训练，但大多数研究者并不会自己构建网页爬虫来获取训练数据。相反，他们使用公开可用的爬取数据集。最流行的公开网页爬取数据集来自 Common Crawl——一个非营利组织，提供免费的网页语料库，"覆盖超过 2500 亿页面，时间跨度达 17 年"。

然而，将 Common Crawl（CC）的原始数据转换为可用的语言模型训练数据需要大量工作。例如，网页的原始数据是 HTML 格式，我们需要从中提取文本。此外，许多页面可能质量低劣、是精确或近似的重复内容、包含有害内容，或含有敏感信息，我们需要过滤掉这些页面，或删除其内容中的不良部分。在本作业中，我们将建立一条完成上述若干步骤的流水线，将原始互联网数据转化为可用于语言模型训练的数据集。

### 2.1 观察数据

在实现任何功能之前，观察原始数据并对其有直观认识总是很有价值的。CC 数据有三种格式：

**WARC**

（"Web ARChive format"）文件包含原始 CC 数据，涵盖页面 ID 和 URL、元数据和 HTTP 请求详情（如请求的日期和时间、服务器 IP 地址），以及页面的原始内容（如 HTML）。

**WAT**

（"Web Archive Transformation"）文件包含更高层次的元数据，从 WARC 文件中提取并以 JSON 对象形式转储。例如，对于 HTML 页面，包含该页面的链接列表和页面标题。

**WET**

（"Web Extracted Text"）文件包含从原始 HTML 页面提取的纯文本。

对于以下问题，我们将查看一个 WARC 文件及其对应的 WET 文件。这些文件来自 2026 年的一次 Common Crawl 爬取，下载方式如下：

```bash
# 下载一个 WARC 文件样例
$ wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/segments/1741613651012.6/warc/CC-MAIN-20260310152245-20260310182245-00000.warc.gz
# 下载其对应的 WET 文件
$ wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/segments/1741613651012.6/wet/CC-MAIN-20260310152245-20260310182245-00000.warc.wet.gz
```

这些文件可在以下路径获取：

- `/shared-data/CC/example.warc.gz`
- `/shared-data/CC/example.warc.wet.gz`

> **警告**：这些文件包含完全未经过滤的互联网页面，可能含有大量潜在的有害内容。如果你看到不想阅读的文档，可以直接跳过。

---

> **问题（`look_at_cc`）：观察 Common Crawl（4 分）**
>
> (a) 使用上述路径提供的 WARC 文件副本。我们来看这个文件中的第一个页面。这是一个 gzip 压缩文件，可以用以下命令浏览其内容：
>
> ```bash
> $ zcat /shared-data/CC/example.warc.gz | less
> ```
>
> `less` 允许你用键盘方向键、Page Up、Page Down 浏览文件。按 `q` 退出。
>
> 查看最开头的那个网页。它的 URL 是什么？它现在还能访问吗？从原始 HTML 来看，你能判断这个页面大致是关于什么内容的吗？
>
> **交付物**：2-3 句话的回答。
>
> (b) 现在来看对应的 WET 文件：
>
> ```bash
> $ zcat /shared-data/CC/example.warc.wet.gz | less
> ```
>
> 注意 WET 文件包含 HTTP 头信息（如 `Content-Length`），这些并不属于提取文本的内容。查看第一个样例，你会看到它包含了从你刚才看到的原始 HTML 中提取的文本。
>
> 注意提取的文本中有很多内容让人联想到 HTML 的结构，而不是页面的主要内容。你认为提取器应该过滤掉哪些部分？思考这类文本作为训练数据的质量：在这类文本上训练模型会出现什么问题？相反，模型能从这个页面中提取哪些有用信息？
>
> **交付物**：3-4 句话的回答。
>
> (c) 什么是好的训练样本，这在很大程度上取决于具体场景。描述一个这个样例可能有用的应用领域，以及一个可能没用的应用领域。
>
> **交付物**：1-2 句话的回答。
>
> (d) 再多看 25 条 WET 记录，以更好地了解 Common Crawl 的内容构成。对每条记录，简短说明文档的语言（如果你能识别的话）、域名、页面类型等。在你看到多少个样例之后才出现了你认为"高质量"的网页？
>
> **交付物**：对 25 个文档的简短标注，包括语言、域名、页面类型及其他杂项说明，以及出现高质量样例所需的样例数。

---

### 2.2 HTML 转文本

正如你从上面的 WARC 和 WET 文件中注意到的，从 HTML 中提取文本是一项挑战。通常，任何提取过程都会寻找 HTML 中的可见内容（如 `<p>` 标签，它应该包含文本块），但这往往会提取出比我们在浏览器中打开页面时所感知到的主要内容多得多的内容。例如，打开 StackOverflow 时，主要内容在问题和回答中，但技术上来说，菜单选项、其他 StackExchange 站点的链接、页脚、登录/注册链接——这些都是可见文本，从这些内容中可靠地区分页面主要内容是很有挑战性的。

许多工具实现了文本提取流水线。在本作业中，我们使用 [Resiliparse](https://resiliparse.chatnoir.eu/en/stable/index.html) 库进行文本提取。Resiliparse 还能解决一个更基础的问题：检测包含原始内容的字节的文本编码。虽然网络上大多数页面采用 UTF-8 编码（根据维基百科，占比 98.2%），但我们的文本提取流水线应该对其他编码也保持健壮。

**注意：**

我们建议使用 [FastWARC](https://resiliparse.chatnoir.eu/en/stable/index.html) 库来迭代每个 WARC 文件中的记录。具体来说，以下类可能有用：

```python
from fastwarc.warc import ArchiveIterator, WarcRecordType
```

---

> **问题（`extract_text`）：HTML 转文本（3 分）**
>
> (a) 编写一个函数，从包含原始 HTML 的字节串中提取文本。使用 `resiliparse.extract.html2text.extract_plain_text` 执行提取。该函数需要一个字符串，因此你需要先将字节串解码为 Unicode 字符串。注意输入的字节串不一定是 UTF-8 编码，所以当 UTF-8 解码失败时，你的函数应该能够检测编码。`resiliparse.parse.encoding.detect_encoding()` 可能对此有用。
>
> **交付物**：一个接受包含 HTML 的字节串并返回包含提取文本的字符串的函数。实现适配器 [`run_extract_text_from_html_bytes`] 并确保通过 `uv run pytest -k test_extract_text_from_html_bytes`。
>
> (b) 在单个 WARC 文件上运行你的文本提取函数。将其输出与对应 WET 文件中的提取文本进行比较。你注意到了哪些差异和/或相似之处？哪种提取效果更好？
>
> **交付物**：2-3 句话，对比你自己的函数与 WET 文件中提取文本的异同。

---

### 2.3 语言识别

网络上包含数千种语言的页面。但在大多数计算预算下，训练一个能有效利用如此多样化数据的多语言模型是很有挑战性的。因此，许多基于 Common Crawl 的语言模型训练集只包含有限几种语言的数据。

[fastText](https://fasttext.cc) 是一个用于此目的的实用库，它提供高效的文本分类器。该库既提供训练你自己分类器的基础设施，也提供一系列预训练模型，包括用于语言识别的模型。你可以从 `https://fasttext.cc/docs/en/language-identification.html` 下载 fastText 语言识别模型 `lid.176.bin`；它也可在 `/shared-data/classifiers/lid.176.bin` 获取。

通常，语言过滤器使用分类器给出的分数来决定是否保留某个页面。使用 fastText 语言识别分类器实现一个语言识别过滤器，该过滤器应给出一个非负分数，表示其对预测结果的置信度。

---

> **问题（`language_identification`）：语言识别（6 分）**
>
> (a) 编写一个函数，接受一个 Unicode 字符串，识别该字符串中主要使用的语言。你的函数应返回一个二元组，包含语言的标识符以及表示该预测置信度的 0 到 1 之间的分数。
>
> **交付物**：一个执行语言识别的函数，给出最可能的语言预测及分数。实现适配器 [`run_identify_language`] 并确保通过 `uv run pytest -k test_identify_language`。注意这些测试假定英语的字符串标识符为 `"en"`，中文为 `"zh"`，因此如果需要的话，你的测试适配器应执行相应的重映射。
>
> (b) 语言模型在推理时的行为很大程度上取决于其训练数据。因此，数据过滤流水线中的问题可能会导致下游问题。你认为语言识别过程中的问题可能导致哪些问题？在更高风险的场景下（例如部署面向用户的产品时），你会如何缓解这些问题？
>
> **交付物**：2-5 句话的回答。
>
> (c) 在通过你之前实现的文本提取函数从 WARC 文件中提取的文本上运行你的语言识别系统。手动识别 20 个随机样本的语言，并与分类器的预测进行比较。报告分类器的任何错误。英文文档占多大比例？根据你的观察，在过滤时使用什么样的分类器置信度阈值比较合适？
>
> **交付物**：2-5 句话的回答。

---

### 2.4 个人可识别信息

网络上存在大量可用于联系或识别个人的信息，如电子邮件地址、电话号码或 IP 地址。我们可能不希望面向用户的语言模型输出关于真实人物的此类信息，因此屏蔽训练数据集中的这些信息是一个常见步骤。

你现在将实现三个程序，分别用于屏蔽 (a) 电子邮件地址、(b) 电话号码和 (c) IP 地址。

---

> **问题（`mask_pii`）：个人可识别信息（3 分）**
>
> (a) 编写一个函数来屏蔽电子邮件地址。你的函数接受一个字符串作为输入，将所有电子邮件地址实例替换为字符串 `"|||EMAIL_ADDRESS|||"`。要可靠地检测电子邮件地址，可以查找能做到这一点的正则表达式。
>
> **交付物**：一个将给定字符串中所有电子邮件地址替换为 `"|||EMAIL_ADDRESS|||"` 的函数，返回包含新字符串和被屏蔽实例数量的二元组。实现适配器 [`run_mask_emails`] 并确保通过 `uv run pytest -k test_mask_emails`。
>
> (b) 编写一个函数来屏蔽电话号码。你的函数接受一个字符串作为输入，将所有电话号码实例替换为字符串 `"|||PHONE_NUMBER|||"`。可靠地做到这一点极具挑战性，因为电话号码的书写格式多种多样，但你至少应该捕获美国最常见的电话号码格式，并对轻微的语法偏差保持健壮。
>
> **交付物**：一个将给定字符串中所有电话号码替换为 `"|||PHONE_NUMBER|||"` 的函数，返回包含新字符串和被屏蔽实例数量的二元组。实现适配器 [`run_mask_phone_numbers`] 并确保通过 `uv run pytest -k test_mask_phones`。
>
> (c) 编写一个函数来屏蔽 IP 地址。对于这个问题，只需关注 IPv4 地址（用点分隔的 4 个 0 到 255 之间的数字）即可。你的函数接受一个字符串作为输入，将所有 IP 地址实例替换为字符串 `"|||IP_ADDRESS|||"`。
>
> **交付物**：一个将给定字符串中所有 IPv4 地址替换为 `"|||IP_ADDRESS|||"` 的函数，返回包含新字符串和被屏蔽实例数量的二元组。实现适配器 [`run_mask_ips`] 并确保通过 `uv run pytest -k test_mask_ips`。
>
> (d) 当这些过滤器被朴素地应用于训练集时，你认为下游语言模型可能会出现哪些问题？你会如何缓解这些问题？
>
> **交付物**：2-5 句话的回答。
>
> (e) 在通过你之前实现的文本提取函数从 WARC 文件中提取的文本上运行你的 PII 屏蔽函数。查看 20 个发生了替换的随机样本；给出一些假阳性和假阴性的例子。
>
> **交付物**：2-5 句话的回答。

---

### 2.5 有害内容

来自网络的未经过滤的数据转储中含有大量我们不希望语言模型在推理时复述的文本。其中一些训练样本甚至可能来自看似无害的网站，如维基百科——例如，某些页面上用户留下的评论可能相当有毒。虽然要建立关于"有害"的明确界定几乎是不可能的，但许多数据过滤流水线仍会对主要含有有害内容的页面进行一定程度的过滤。

识别此类内容的方法有很多，包括统计禁用词列表中词语的出现次数，或基于人工标注者提供的标签构建简单的分类器。在本作业的这一部分，我们将重点识别两大类有害内容：NSFW（"Not Safe For Work"，包括色情、亵渎或其他可能令人不安的内容）和有毒言论（"粗鲁、不尊重或不合理的、可能使人离开讨论的语言"）。我们将使用 Dolma [1] 项目提供的 fasttext 预训练模型来判断一段文本是否属于这两类内容之一。这些分类器在 Jigsaw 毒性评论数据集上训练，该数据集包含在多种标签下分类的维基百科评论。

NSFW 分类器可在以下地址下载：

`dolma-artifacts.org/.../jigsaw_fasttext_bigrams_nsfw_final.bin`

仇恨言论分类器可在以下地址下载：

`dolma-artifacts.org/.../jigsaw_fasttext_bigrams_hatespeech_final.bin`

这两个分类器也已放置在共享数据目录中：

- `/shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin`：仇恨言论和有毒言论的预训练分类器
- `/shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin`：NSFW 内容的预训练分类器

使用这些模型实现一个函数，接受包含页面内容的 Unicode 字符串，并返回一个标签（如 `"toxic"`、`"non-toxic"`），以及一个置信度分数。

---

> **问题（`harmful_content`）：有害内容（6 分）**
>
> (a) 编写一个函数来检测 NSFW 内容。
>
> **交付物**：一个函数，将给定字符串标记为是否包含 NSFW 内容，返回包含标签和置信度分数的二元组。实现适配器 [`run_classify_nsfw`] 并确保通过 `uv run pytest -k test_classify_nsfw`。注意这个测试只是一个合理性检查，取自 Jigsaw 数据集，但并不断言你的分类器是准确的——这一点你应该自行验证。
>
> (b) 编写一个函数来检测有毒言论。
>
> **交付物**：一个函数，将给定字符串标记为是否包含有毒言论，返回包含标签和置信度分数的二元组。实现适配器 [`run_classify_toxic_speech`] 并确保通过 `uv run pytest -k test_classify_toxic_speech`。同样，这个测试只是一个合理性检查，也取自 Jigsaw 数据集。
>
> (c) 当这些过滤器被应用于训练集时，你认为下游语言模型可能会出现哪些问题？你会如何缓解这些问题？
>
> **交付物**：2-5 句话的回答。
>
> (d) 在通过你之前实现的文本提取函数从 WARC 文件中提取的文本上运行你的有害内容过滤器。查看 20 个随机样本，并将分类器的预测与你自己的判断进行比较。报告任何分类器错误。有多少比例的文档是有害的？根据你的观察，在过滤时使用什么样的分类器置信度阈值比较合适？
>
> **交付物**：2-5 句话的回答。

---

### 2.6 质量规则

即使按语言过滤页面并删除有害内容之后，仍然有相当大比例的页面对语言模型训练来说质量太低。同样，"质量"并不容易定义，但浏览 Common Crawl 样本有助于识别低质量内容的典型例子，例如：

- 有付费墙的页面
- 损坏链接的占位页面
- 登录、注册或联系表单
- 主要包含非文本内容的页面，这些内容在文本提取时会丢失（如照片、视频）

Gopher 论文 [2] 描述了一组简单的质量过滤器，用于从网页爬取数据中删除类似的简单低质量文本案例。这些过滤器由简单的启发式规则组成，易于理解，通常能覆盖许多显而易见的不合适样本。Gopher 质量过滤器基于文档长度、词长、符号-词比率以及某些英文停用词的出现频率等标准。对于本作业，你将实现 Gopher 论文 [2] 中描述的过滤器子集。具体来说，你应该删除满足以下任一条件的文档：

- 包含少于 50 个或多于 100,000 个单词。
- 平均词长不在 3 到 10 个字符的范围内。
- 超过 30% 的行以省略号（`"..."`）结尾。
- 含有至少一个字母字符的单词占比不足 80%。

关于 Gopher 论文使用的所有质量过滤器的完整描述，请参考其附录 A。

---

> **问题（`gopher_quality_filters`）：Gopher 质量过滤器（3 分）**
>
> (a) 实现（至少）上述描述的 Gopher 质量过滤器子集。对文本进行分词时，你可能会发现 NLTK 包有用（具体来说是 `nltk.word_tokenize`），但并非必须使用。
>
> **交付物**：一个以字符串作为唯一参数、返回布尔值指示文本是否通过 Gopher 质量过滤器的函数。实现适配器 [`run_gopher_quality_filter`]。然后确保你的过滤器通过 `uv run pytest -k test_gopher` 中的测试。
>
> (b) 在通过你之前实现的文本提取函数从 WARC 文件中提取的文本上运行你的基于规则的质量过滤器。查看 20 个随机样本，并将过滤器的预测与你自己的判断进行比较。评论任何过滤器判断与你判断不一致的案例。
>
> **交付物**：2-5 句话的回答。

---

### 2.7 质量分类器

现在让我们超越 Gopher 规则所捕捉的简单语法标准。语言模型训练并不是第一个需要按质量对内容排序的应用场景——实际上，文本质量排序也是信息检索的基本挑战之一。搜索引擎利用的一个经典信号是网络上的链接结构：高质量页面往往会链接到其他高质量页面 [3]。OpenAI 在构建 WebText（GPT-2 [4] 的训练数据集）时使用了类似的洞察：他们收集了 Reddit 评论中链接数量超过最低"karma"阈值的页面。替代 Reddit 的一个方案是使用维基百科作为高质量链接的来源，因为被维基百科页面外链引用的外部资源往往是可信任的页面 [5]。

使用受控来源通常能产生高质量内容，但由此得到的数据集规模对于当前标准来说太小（OpenWebText 有 40GB 文本，而 The Pile 比它大 20 倍以上）。一个解决方案是将这些参考页面作为正例，将（随机的）Common Crawl 页面作为负例，训练一个 fastText 分类器。该分类器给出一个质量分数，你可以用它来从整个 Common Crawl 中过滤页面。设置质量阈值需要在精确率和召回率之间权衡。

在本作业的这一部分，你将构建一个质量分类器。为方便起见，课程组已从最近的维基百科数据转储中提取了参考页面的 URL，并将其放置在共享数据目录中，路径为 `/shared-data/wiki/enwiki-20260501-extracted_urls.txt.gz`。该文件包含 2026 年 5 月英文维基百科页面中找到的外部链接，但我们希望你对这些 URL 进行子采样，

以获取用于训练分类器的"高质量"文本正例。注意这些正例可能仍包含不良内容，因此可能有必要对它们应用你构建的其他过滤原语，以进一步提高质量。给定一个 URL 文件，你可以用 `wget` 以 WARC 格式爬取其内容：

```bash
wget --timeout=5 \
    -i subsampled_positive_urls.txt \
    --warc-file=subsampled_positive_urls.warc \
    -O /dev/null
```

---

> **问题（`quality_classifier`）：质量分类器（15 分）**
>
> (a) 训练一个质量分类器，给定文本，返回一个数值质量分数。
>
> **交付物**：一个用于下一个子问题的质量分类器。
>
> (b) 编写一个函数，将页面标记为高质量或低质量，并提供标签的置信度分数。
>
> **交付物**：一个以字符串作为唯一参数、返回包含标签（高质量或否）和置信度分数的二元组的函数。实现适配器 [`run_classify_quality`]。作为合理性检查，确保它通过 `uv run pytest -k test_classify_quality` 正确分类我们提供的两个样例。

---

## 3 去重

网络上存在大量重复内容。有些页面是彼此的精确副本——想想网站存档，或由标准工具生成的默认页面，如流行 Web 服务器的 404 页面。但大部分重复发生在更细粒度的层次上。例如，考虑 Stack Overflow 上所有问题页面：虽然每个页面都有独特内容（问题、评论、回答本身），但所有页面都有大量冗余内容，如页眉、菜单选项和页脚，当所有这些页面被渲染时，这些内容会以完全相同的形式出现。在本节的第一部分，我们将处理这种精确重复的情况，之后再看如何处理近似重复。

### 3.1 精确行去重

一种对精确重复进行去重的简单方法是只保留文档中在语料库中唯一的行。事实证明，这足以消除大部分冗余，例如我们上面提到的页眉和菜单选项。在较简单的情况下，删除在其他地方精确重复的行之后，我们通常只剩下每个页面的唯一主要内容（如 StackOverflow 上的问题和回答）。

为此，我们可以对语料库进行一次遍历，统计观察到的每一行的出现次数。然后在第二次遍历中，通过只保留唯一的行来重写每个文档。

朴素地看，保存计数器的数据结构可能会占用存储语料库中所有唯一行所需的全部空间。一个简单的内存效率技巧是改用行的哈希值作为键，使键具有固定大小（而不是依赖行的长度）。你现在将实现这种简单的去重方法。

---

> **问题（`exact_deduplication`）：精确行去重（3 分）**
>
> 编写一个函数，接受输入文件路径列表，对其执行精确行去重。它应该首先使用哈希来统计语料库中每行的频率以减少内存占用，然后通过只保留唯一行来重写每个文件。
>
> **交付物**：一个执行精确行去重的函数。你的函数应接受两个参数：(a) 输入文件路径列表，(b) 输出目录。它应将每个输入文件重写到与原文件同名的输出目录中，但通过删除在输入文件集合中出现超过一次的行来进行去重。例如，如果输入路径为 `a/1.txt` 和 `a/2.txt`，输出目录为 `b/`，你的函数应写出文件 `b/1.txt` 和 `b/2.txt`。实现适配器 [`run_exact_line_deduplication`] 并确保通过 `uv run pytest -k test_exact_line_deduplication`。

---

### 3.2 MinHash + LSH 文档去重

精确去重有助于删除跨多个网页逐字重复的内容，但无法处理文档内容略有不同的情况。例如，考虑软件许可证文件——许可证文件通常是由需要填写年份和软件作者姓名的模板生成的。因此，一个采用 MIT 许可证的项目的许可证文件与另一个采用 MIT 许可证的项目内容基本相同，但它们并不是*精确*的副本。为了去除这种重复的、大量模板化的内容，我们需要模糊去重，并且为了高效地进行文档级模糊去重，我们将使用 MinHash 结合局部敏感哈希（LSH）。

为了执行模糊去重，我们将使用文档间的特定相似度度量：文档 n-gram 的 Jaccard 相似度。集合 $S$ 和 $T$ 之间的 Jaccard 相似度定义为：

$$
J(S, T) = \frac{|S \cap T|}{|S \cup T|}
$$

要朴素地执行模糊去重，我们可以将每个文档表示为一组 n-gram，并计算所有文档对之间的 Jaccard 相似度，将相似度超过特定阈值的对标记为重复。然而，这种方法对大型文档集合（如 Common Crawl）来说是不可行的。此外，朴素地存储一组 n-gram 将占用比文档本身多得多的内存。

#### MinHash

为解决内存问题，我们用*签名*（signature）替代 n-gram 文档表示。特别地，我们希望构建这样的签名：如果我们比较两个文档的签名，就能近似得到两个文档各自 n-gram 集合之间的 Jaccard 相似度。MinHash 签名满足这些特性。

为了对一组文档 n-gram $S = \{s_1, \ldots, s_n\}$ 计算 MinHash 签名，我们需要 $k$ 个不同的哈希函数 $h_1, \ldots, h_k$，每个哈希函数将一个 n-gram 映射到一个整数。给定哈希函数 $h_i$，n-gram 集合 $S$ 的 minhash 为：

$$
\text{minhash}(h_i, S) = \min\bigl(h_i(s_1),\, h_i(s_2),\, \ldots,\, h_i(s_n)\bigr)
$$

文档 n-gram 集合 $S$ 的签名是 $\mathbb{R}^k$ 中的一个向量，其中第 $i$ 个元素包含在随机哈希函数 $h_i$ 下 $S$ 的 minhash，即：

$$
\bigl[\text{minhash}(h_1, S),\, \text{minhash}(h_2, S),\, \ldots,\, \text{minhash}(h_k, S)\bigr]
$$

结果表明，对于两个文档的 n-gram 集合 $S_1$ 和 $S_2$，这些集合之间的 Jaccard 相似度可以由签名中具有相同 minhash 值的列的比例来近似（证明参见 [6] 第 3 章 3.3.3 节）。例如，给定文档签名 $[1, 2, 3, 2]$ 和 $[5, 2, 3, 4]$，Jaccard 相似度近似为 $2/4$，因为这两个签名的第二列和第三列具有相同的 minhash 值。

#### 局部敏感哈希（LSH）

虽然 MinHash 给了我们一种内存高效的文档表示，能保持任意文档对之间的期望相似度，但我们仍然需要比较所有文档对来找出相似度最高的那些。LSH 提供了一种高效地将可能具有高相似度的文档分桶的方法。为了将 LSH 应用于我们的文档签名（现在是 $\mathbb{R}^k$ 中的向量），我们将签名分成 $b$ 个条带，每个条带包含 $r$ 个 minhash，其中 $k = br$。例如，如果我们有 100 元素的文档签名（由 100 个随机哈希函数生成），我们可以将其分成 2 个各含 50 个 minhash 的条带，或 4 个各含 25 个 minhash 的条带，或 50 个各含 2 个 minhash 的条带，等等。如果两个文档在特定条带中具有相同的哈希值，它们将被聚集到同一桶中，并被视为候选重复对。因此，对于固定数量的签名，增加条带数会提高召回率并降低精确率。

举一个具体的例子，假设我们有文档 $D_1$，其 minhash 签名为 $[1, 2, 3, 4, 5, 6]$，以及另一个文档 $D_2$，其 minhash 签名为 $[1, 2, 3, 5, 1, 2]$。如果我们使用 3 个条带，每个条带含 2 个 minhash，则 $D_1$ 的第一个条带为 $[1, 2]$，$D_1$ 的第二个条带为 $[3, 4]$，$D_1$ 的第三个条带为 $[5, 6]$。类似地，$D_2$ 的第一个条带为 $[1, 2]$，$D_2$ 的第二个条带为 $[3, 5]$，$D_2$ 的第三个条带为 $[1, 2]$。由于第一个条带中的哈希值匹配（$D_1$ 和 $D_2$ 的第一个条带都是 $[1, 2]$），$D_1$ 和 $D_2$ 将在第一个条带下被聚集在一起。它们不会在其他任何条带下被聚集，因为那些条带的哈希值不匹配。然而，由于文档在至少一个条带下被聚集，无论其他条带是否匹配，它们都会被视为候选重复对。

一旦我们找到了候选重复对，就可以用多种方式处理它们。例如，我们可以计算所有候选重复对之间的 Jaccard 相似度，并将超过设定阈值的对标记为重复。

最后，我们对各桶中的重复文档进行聚类。例如，假设文档 A 和 B 在某个桶中匹配，且其真实 Jaccard 相似度高于我们的阈值；文档 B 和 C 在另一个桶中匹配，且其真实 Jaccard 相似度也高于我们的阈值。那么我们将 A、B、C 视为一个单独的簇，每个簇中随机删除所有文档，只保留一个。

---

> **问题（`minhash_deduplication`）：MinHash + LSH 文档去重（8 分）**
>
> 编写一个函数，接受输入文件路径列表，用 MinHash 和 LSH 执行模糊文档去重。具体来说，你的函数应为提供的路径列表中的每个文档计算 minhash 签名，使用 LSH 和提供的条带数来识别候选重复对，然后计算候选重复对之间真实的 n-gram Jaccard 相似度，并删除超过给定阈值的文档。为了提高召回率（参照 [7]），在计算 minhash 签名和/或比较 Jaccard 相似度之前，通过转小写、删除标点、规范化空白、删除变音符号和应用 NFD Unicode 规范化来对文本进行归一化。
>
> **交付物**：一个执行模糊文档去重的函数。你的函数至少应接受以下参数：(a) 输入文件路径列表，(b) 用于计算 minhash 签名的哈希函数数量，(c) 用于 LSH 的条带数，(d) 用于计算 minhash 签名的 n-gram 长度（以词为单位），(e) 输出目录。你可以假设用于计算 minhash 签名的哈希函数数量可以被用于 LSH 的条带数整除。
>
> 你的函数应将每个输入文件重写到输出目录，文件名相同，但只写出 (a) 不是候选重复的文档，和/或 (b) 从聚集桶中随机选择保留的文档。例如，如果输入路径为 `a/1.txt` 和 `a/2.txt`，输出目录为 `b/`，你的函数应写出文件 `b/1.txt` 和 `b/2.txt`。实现适配器 [`run_minhash_deduplication`] 并确保通过 `uv run pytest -k test_minhash_deduplication`。

---

## 4 过滤语言模型训练数据

现在我们已经实现了各种过滤网页爬取数据的原语，让我们实际使用它们来生成一些语言模型训练数据。

在本作业的这一部分，你的目标是过滤一批 CC WET 文件，以产出语言模型训练数据。你将从已经过英文过滤的 WET 文件开始：我们保留了英文概率至少为 70%（根据 fastText `lid.176.bin` 语言识别模型）的文档。我们已将这些经英文过滤的 WET 文件放置在 `/shared-data/english-wet-data` 下供你使用。

具体来说，你的目标是过滤 CC 数据转储，创建语言模型训练数据，使在该数据上训练的 Transformer 语言模型在 Paloma 基准 [8] 的 C4 100 个领域子集上的验证困惑度最小化。**你不应修改模型架构或训练流程**，因为目标是构建最优的*数据*。该数据集包含来自 C4 语言模型数据集 [9] 100 个最常见领域的样本。课程组已将该数据的一个副本放置在 `/shared-data/tokenized_paloma_c4_100_domains_validation.bin`（使用 GPT-2 分词器进行分词）——你可能会发现查看这些数据有助于了解其内容。你可以用以下方式加载它：

```python
import numpy as np
data = np.fromfile(
    "/shared-data/tokenized_paloma_c4_100_domains_validation.bin",
    dtype=np.uint16
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.decode(data[0:2000]))
```

给定你过滤后的数据集，你将在该数据上训练一个 GPT-2 小形状的模型，并评估其在 C4 100 上的困惑度。

**我们注意到，你*可以*在构建过滤器或分类器来处理 CC WET 文件时使用 Paloma 验证数据，但不允许将任何验证数据逐字复制到你的训练数据中。语言模型绝不应见到任何来自验证集的数据。**

即使这些 WET 文件已经是大量数据，为了高效处理这些数据，我们建议尽可能使用多进程。具体来说，你可能会发现 Python 的 `concurrent.futures` 或 `multiprocessing` API 很有帮助。下面是一个使用 `concurrent.futures` 跨多个进程并行化函数的简单示例：

```python
import concurrent.futures
import os

from tqdm import tqdm

def process_single_wet_file(input_path: str, output_path: str):
    # TODO: 读取 input_path，处理输入，将输出写入 output_path
    return output_path

# 配置 executor
num_cpus = len(os.sched_getaffinity(0))
executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
wet_filepaths = ["a.warc.wet.gz", "b.warc.wet.gz", "c.warc.wet.gz"]
output_directory_path = "/path/to/output_directory/"

futures = []
for wet_filepath in wet_filepaths:
    # 对每个 warc.wet.gz 文件，向 executor 提交一个任务并获取 future
    wet_filename = str(pathlib.Path(wet_filepath).name)
    future = executor.submit(
        process_single_wet_file,
        wet_filepath,
        os.path.join(output_directory_path, wet_filename)
    )
    # 保存 future
    futures.append(future)

# 随着 future 完成，用进度条迭代已完成的 future
for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(wet_filepaths),
):
    output_file = future.result()
    print(f"Output file written: {output_file}")
```

在 Modal 上使用本地并行可使用 `concurrent.futures` 或 `multiprocessing`，跨 worker 并行则在你的输入上调用 `.map(...)`。

我们还建议使用 [FastWARC](https://resiliparse.chatnoir.eu/en/stable/index.html) 库来迭代每个 WET 文件中的记录，以及 [tldextract](https://github.com/john-kurkowski/tldextract) 库从 URL 中提取域名进行过滤。具体来说，以下类可能有用：

```python
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract
```

---

> **问题（`filter_data`）：过滤语言模型训练数据（6 分）**
>
> (a) 编写脚本，从 Common Crawl WET 文件（位于 `/shared-data/english-wet-data`）中过滤语言模型训练数据。你可以自由使用我们在作业早期部分实现的任何原语，也可以探索其他过滤方法（例如，基于 n-gram 语言模型困惑度进行过滤）。你的目标是生成能够最小化 C4 100 个领域子集的 Paloma 基准困惑度的数据。
>
> **再次提醒，你*可以*在构建过滤器或分类器来处理 CC WET 文件时使用 Paloma 验证数据，但不允许将任何验证数据逐字复制到你的训练数据中。**
>
> 你的脚本应报告各过滤步骤保留的样本数量，以便了解各过滤步骤对最终输出数据的贡献。
>
> **交付物**：一个（或一系列）并行过滤所提供 CC WET 文件以产出语言模型训练数据的脚本，以及对各过滤步骤删除样本比例的书面分析。
>
> (b) 过滤提供的 WET 文件（原始 2,500 个 WET 文件）需要多长时间？过滤完整的 Common Crawl 数据转储需要多长时间？
>
> **交付物**：数据过滤流水线的运行时间。

---

现在我们已经生成了一些语言模型训练数据，让我们仔细看看，以更好地理解其内容。

---

> **问题（`inspect_filtered_data`）：检视过滤数据（4 分）**
>
> (a) 从你过滤后的数据集中随机抽取 5 个样本。评论它们的质量以及是否适合用于语言模型训练，尤其是考虑到我们的目标是最小化 C4 100 个领域基准的困惑度。只展示较长文档的相关摘录即可。
>
> **交付物**：来自最终过滤数据的 5 个随机样本，对每个样本给出 1-2 句描述：文档内容如何，是否值得用于语言模型训练。
>
> (b) 抽取 5 个被你的过滤脚本删除和/或修改的 CC WET 文件样本。是你的过滤流程的哪个部分删除或修改了这些文档，你认为删除/修改是否合理？
>
> **交付物**：来自原始 WET 的 5 个随机丢弃样本，对每个样本给出 1-2 句描述：文档内容如何，删除是否合理。
>
> (c) 如果你上面的分析启发你对数据流水线进行进一步修改，可以在训练模型前自由地进行这些修改。报告你尝试的任何数据修改和/或数据迭代。
>
> **交付物**：对你尝试的数据修改和/或数据迭代的描述。

---

在用我们的数据训练语言模型之前，我们需要对其进行分词。使用 `transformers` 提供的 GPT-2 分词器将你过滤后的数据编码为整数 ID 序列，用于训练语言模型。不要忘记在每个文档后面添加 GPT-2 的 end-of-sequence token `<|endoftext|>`。以下是一些起始代码：

```python
import multiprocessing

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

input_path = "path/to/your/filtered/data"
output_path = "path/to/your/tokenized/data"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_line_and_add_eos(line):
    return tokenizer.encode(line) + [tokenizer.eos_token_id]

with open(input_path) as f:
    lines = f.readlines()

pool = multiprocessing.Pool(multiprocessing.cpu_count())
chunksize = 100
results = []
for result in tqdm(
    pool.imap(tokenize_line_and_add_eos, lines, chunksize=chunksize),
    total=len(lines),
    desc="Tokenizing lines"
):
    results.append(result)

pool.close()
pool.join()

# 展平 ID 列表并转换为 numpy 数组
all_ids = [token_id for sublist in results for token_id in sublist]
print(f"Tokenized and encoded {input_path} into {len(all_ids)} tokens")
ids_array = np.array(all_ids, dtype=np.uint16)
ids_array.tofile(output_path)
```

---

> **问题（`tokenize_data`）：分词（2 分）**
>
> 编写脚本，对你过滤后的数据进行分词和序列化。确保按照上述示例代码进行序列化，使用 `ids_array.tofile(output_path)`，其中 `ids_array` 是 `np.uint16` 格式的整数 ID numpy 数组。这确保了与提供的训练脚本的兼容性。
>
> 你过滤后的数据集中共有多少个 token？
>
> **交付物**：分词和序列化你过滤后数据的脚本，以及产出数据集中的 token 数量。

---

现在我们已经对数据进行了分词，可以用它来训练模型了。我们将在生成的数据上训练一个 GPT-2 小形状、约 430M 参数的模型，定期在 C4 100 个领域数据集上测量验证性能。训练运行使用 8 块 B200 GPU、数据并行、每设备 batch size 128、上下文长度 512、共 16,384 步优化。在上下文长度 512 的情况下，这相当于训练期间采样的约

$$
2^{33} = 8.6\text{B}
$$

个训练 token。使用这些设置，课程组的训练运行在约 2 小时内完成。

使用 `scripts/train.py` 中的训练脚本启动训练（你将使用的超参数见 `cs336_basics/train_config.py`）。用 `--train-bin` 传入你的 tokenized 训练数据路径：

```bash
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin
```

请再次注意，作业的目标是通过优化*数据*来最小化验证损失，而不是通过修改模型和/或优化流程，因此**不要修改训练流程或训练脚本**。

训练完成后，你可以用以下命令从保存的模型中采样：

```bash
uv run modal run scripts/generate_with_gpt2_tok.py --model-path /root/data/output/your_data
```

---

> **问题（`train_model`）：训练模型（8 分）**
>
> 在你的 tokenized 数据集上训练一个语言模型（GPT-2 小形状，约 430M 参数）。定期测量在 C4 100 个领域上的验证损失，你的模型取得的最优验证损失是多少？
>
> **交付物**：记录的最优验证损失、对应的学习曲线，以及对你所做工作的描述。

---

## 5 参考文献

[1] L. Soldaini *et al.*, "Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research," *arXiv preprint arXiv:2402.00159*, 2024.

[2] J. W. Rae *et al.*, "Scaling language models: Methods, analysis & insights from training gopher," *arXiv preprint arXiv:2112.11446*, 2021.

[3] L. Page, S. Brin, R. Motwani, and T. Winograd, "The pagerank citation ranking: Bring order to the web," in *Proc. of the 7th International World Wide Web Conf.*, 1998.

[4] A. Radford *et al.*, "Language models are unsupervised multitask learners," *OpenAI blog*, vol. 1, no. 8, p. 9, 2019.

[5] H. Touvron *et al.*, "LLaMA: Open and Efficient Foundation Language Models," 2023.

[6] J. Leskovec, A. Rajaraman, and J. D. Ullman, *Mining of Massive Datasets*, 2nd ed. USA: Cambridge University Press, 2014.

[7] G. Penedo *et al.*, "The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only," 2023.

[8] I. Magnusson *et al.*, "Paloma: A Benchmark for Evaluating Language Model Fit," 2023.

[9] C. Raffel *et al.*, "Exploring the Limits of Transfer Transfer with a Unified Text-to-Text Transformer," *Journal of Machine Learning Research*, vol. 21, no. 140, pp. 1–67, 2020. [Online]. Available: [http://jmlr.org/papers/v21/20-074.html](http://jmlr.org/papers/v21/20-074.html)
