"""
个人可识别信息（PII）屏蔽模块。

网络爬取数据中存在大量真实的电子邮件、电话号码、IP 地址。
这些信息不应出现在训练数据中，因为语言模型可能在推理时泄露真实用户隐私。

处理策略：用占位符替换，而非删除：
  - 电子邮件 → |||EMAIL_ADDRESS|||
  - 电话号码 → |||PHONE_NUMBER|||
  - IPv4 地址 → |||IP_ADDRESS|||

使用正则表达式实现，每个函数返回 (替换后文本, 替换次数) 二元组。
"""

import re

# ── 电子邮件 ──────────────────────────────────────────────────────────────────
# RFC 5322 的完整版本极为复杂，这里用实用子集覆盖 99%+ 场景：
#   本地部分：字母、数字、点、连字符、加号、下划线
#   域名部分：至少一个点，最后一段 2-6 个字母（TLD）
# 注意：使用负向前瞻/后顾排除已有占位符中的 @（虽然占位符中无 @，保险起见）
_EMAIL_RE = re.compile(
    r"(?<!\|\|\|)"                          # 不在占位符内
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+"  # user@domain
    r"\.[a-zA-Z]{2,6}"                       # .com / .co.jp 等
)

# ── 电话号码（美国格式）────────────────────────────────────────────────────────
# 覆盖以下常见格式：
#   2831823829          → 纯数字 10 位（NXX-NXX-XXXX 无分隔符）
#   283-182-3829        → 连字符分隔（NXX-NXX-XXXX）
#   (283)-182-3829      → 带括号 + 连字符
#   (283) 182 3829      → 带括号 + 空格
#   +1 283-182-3829     → 国际前缀（+1 或 1）
# 用两个分支分别处理：纯 10 位数字 和 带分隔符/括号格式
_PHONE_RE = re.compile(
    r"(?<!\d)"                              # 前面不是数字（防止匹配信用卡号等）
    r"(?:"
    # 分支 1：纯 10 位连续数字（如 2831823829）
    r"\d{10}"
    r"|"
    # 分支 2：带分隔符/括号的格式，可选国际前缀
    r"(?:\+?1[\s.\-]?)?"                    # 可选的国际区号 +1
    r"(?:"
    r"\(\d{3}\)[\s.\-]?"                   # (283) 或 (283)-
    r"|"
    r"\d{3}[\s.\-]"                         # 283- 或 283<空格>
    r")"
    r"\d{3}[\s.\-]?\d{4}"                   # 182-3829 / 182 3829
    r")"
    r"(?!\d)"                               # 后面不是数字
)

# ── IPv4 地址 ──────────────────────────────────────────────────────────────────
# 四段 0-255 的数字，用点分隔
# 每段：0-199 / 200-249 / 250-255
_OCTET = r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
_IPV4_RE = re.compile(
    r"(?<!\d)"                              # 前面不是数字
    rf"(?:{_OCTET}\.){{3}}{_OCTET}"         # x.x.x.x
    r"(?!\d)"                               # 后面不是数字
)

# 占位符常量
_EMAIL_PLACEHOLDER = "|||EMAIL_ADDRESS|||"
_PHONE_PLACEHOLDER = "|||PHONE_NUMBER|||"
_IP_PLACEHOLDER = "|||IP_ADDRESS|||"


def mask_emails(text: str) -> tuple[str, int]:
    """将文本中所有电子邮件地址替换为占位符。

    已有占位符 |||EMAIL_ADDRESS||| 不计入替换次数。

    Returns:
        (替换后文本, 新替换的邮件地址数量)
    """
    count = 0

    def replace(m: re.Match) -> str:
        nonlocal count
        count += 1
        return _EMAIL_PLACEHOLDER

    result = _EMAIL_RE.sub(replace, text)
    return result, count


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """将文本中所有电话号码替换为占位符。

    覆盖美国最常见格式，对轻微语法偏差保持健壮。

    Returns:
        (替换后文本, 新替换的电话号码数量)
    """
    count = 0

    def replace(m: re.Match) -> str:
        nonlocal count
        count += 1
        return _PHONE_PLACEHOLDER

    result = _PHONE_RE.sub(replace, text)
    return result, count


def mask_ips(text: str) -> tuple[str, int]:
    """将文本中所有 IPv4 地址替换为占位符。

    仅处理 IPv4（4 段 0-255），不处理 IPv6。

    Returns:
        (替换后文本, 新替换的 IP 地址数量)
    """
    count = 0

    def replace(m: re.Match) -> str:
        nonlocal count
        count += 1
        return _IP_PLACEHOLDER

    result = _IPV4_RE.sub(replace, text)
    return result, count
