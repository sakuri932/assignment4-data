"""
HTML → 纯文本提取模块。

使用 Resiliparse 库执行文本提取，该库专为大规模网络爬取数据设计，
对畸形 HTML 有极高容错性，底层用 C++ 实现，速度比 BeautifulSoup 快约 10x。

核心思路：
  1. 尝试 UTF-8 解码字节串
  2. 失败时用 Resiliparse 的 detect_encoding 识别编码，再解码
  3. 调用 extract_plain_text 提取主体文本（自动去除脚本、样式、导航等噪声标签）
"""

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """从 HTML 字节串中提取纯文本。

    Args:
        html_bytes: 包含原始 HTML 的字节串，编码不限。

    Returns:
        提取后的纯文本字符串；若无法解码或解析则返回 None。
    """
    # 尝试 UTF-8 解码（覆盖 ~98.2% 的网页）
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # 检测真实编码（如 Latin-1、GBK 等）
        encoding = detect_encoding(html_bytes)
        try:
            html_str = html_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # 极端情况：编码无法识别
            return None

    # Resiliparse 提取主体文本，自动处理 <script>/<style>/<nav> 等噪声
    return extract_plain_text(html_str)
