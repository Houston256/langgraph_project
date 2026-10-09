"""Keeps clickable links in answers to the shop's own domains, also while streaming.

Off-site markdown links keep their text and lose the URL; off-site raw URLs are dropped.
Bare domains without http/www ("moncler.com") stay as plain text: the UI doesn't link them.
"""

import re

from assistant.config import settings

MARKDOWN_LINK = re.compile(
    r"\[([^\]\n]{0,300})\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)"
)
RAW_URL = re.compile(r"<?\b(?:https?://|www\.)[^\s<>()\[\]]+>?", re.IGNORECASE)
# Longest markdown link text worth waiting for before giving up on "[".
MAX_PENDING = 400


def _host(url: str) -> str:
    url = url.strip("<>")
    url = re.sub(r"^(?:https?:)?//", "", url, flags=re.IGNORECASE)
    return url.split("/")[0].split("?")[0].split("#")[0].split(":")[0].lower()


def allowed(url: str) -> bool:
    # Relative and phone links point nowhere else; mail links only to the shop's addresses.
    if url.startswith(("/", "#", "tel:")):
        return True
    host = (
        url[len("mailto:") :].split("@")[-1].split("?")[0].lower()
        if url.startswith("mailto:")
        else _host(url)
    )
    return any(
        host == d or host.endswith("." + d) for d in settings.allowed_link_domains
    )


def sanitize(text: str) -> str:
    """Text with off-site links removed."""
    text = MARKDOWN_LINK.sub(
        lambda m: m.group(0) if allowed(m.group(2)) else m.group(1), text
    )
    # Raw URLs outside markdown links; the markdown ones were handled above.
    out, last = [], 0
    for m in MARKDOWN_LINK.finditer(text):
        out.append(_strip_raw(text[last : m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_strip_raw(text[last:]))
    return "".join(out)


def _strip_raw(text: str) -> str:
    def keep(m: re.Match) -> str:
        url = m.group(0).rstrip(".,;:!?")
        # Sentence punctuation after a dropped URL stays.
        return m.group(0) if allowed(url) else m.group(0)[len(url) :]

    return RAW_URL.sub(keep, text)


class LinkFilter:
    """Streaming version of sanitize for one message: holds back what could still become a link."""

    def __init__(self) -> None:
        self.pending = ""

    def feed(self, delta: str) -> str:
        self.pending += delta
        cut = self._safe_length(self.pending)
        ready, self.pending = self.pending[:cut], self.pending[cut:]
        return sanitize(ready)

    def flush(self) -> str:
        ready, self.pending = self.pending, ""
        return sanitize(ready)

    @staticmethod
    def _safe_length(text: str) -> int:
        safe = len(text)
        # An unfinished "[text](url" might still turn out to be an off-site link.
        start = text.rfind("[")
        if start != -1 and len(text) - start < MAX_PENDING and "\n" not in text[start:]:
            tail = text[start:]
            close = tail.find("]")
            unfinished = (
                close == -1  # "[text" so far
                or close == len(tail) - 1  # "[text]", a "(" may follow
                or (
                    tail[close + 1] == "(" and ")" not in tail[close:]
                )  # "[text](url" so far
            )
            if unfinished:
                safe = start
        # The last word may be a URL in progress: URLs have no whitespace.
        word_start = max(text.rfind(" "), text.rfind("\n"), text.rfind("\t")) + 1
        return min(safe, word_start)
