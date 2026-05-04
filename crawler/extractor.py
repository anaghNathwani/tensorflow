"""
Extract clean text from raw HTML — no external deps, stdlib only.
"""
from html.parser import HTMLParser
import re
import unicodedata


_SKIP_TAGS = {"script", "style", "noscript", "iframe", "svg"}


class _TextExtractor(HTMLParser):
    """Collect all visible text, skip only script/style."""
    def __init__(self):
        super().__init__()
        self._skip = 0
        self._buf:  list[str] = []
        self._paras: list[str] = []
        # Tags that end a paragraph
        self._PARA = {"p","h1","h2","h3","h4","h5","h6",
                      "li","tr","br","div","section","article"}

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        if tag in self._PARA:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        if tag in self._PARA:
            self._flush()

    def handle_data(self, data):
        if self._skip == 0:
            self._buf.append(data)

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        if text:
            self._paras.append(text)
        self._buf = []

    def get_paragraphs(self) -> list[str]:
        self._flush()
        return self._paras


def html_to_text(html: str, min_len: int = 60) -> str:
    """HTML → clean paragraphs. Filters short/noisy lines."""
    html = unicodedata.normalize("NFKC", html)
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass

    seen  = set()
    good  = []
    for para in parser.get_paragraphs():
        # Filter noise: too short, too many brackets/pipes, duplicate
        if len(para) < min_len:
            continue
        if para.count("|") > 5 or para.count("[") > 6:
            continue
        key = para[:80]
        if key in seen:
            continue
        seen.add(key)
        good.append(para)

    return "\n\n".join(good)
