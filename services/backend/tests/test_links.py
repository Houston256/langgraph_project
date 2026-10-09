import random

import pytest

from assistant.runs.links import LinkFilter, allowed, sanitize

CASES = {
    "Najdete je na [stránce prodejen](https://www.myshop.cz/prodejny/).": "Najdete je na [stránce prodejen](https://www.myshop.cz/prodejny/).",
    "Mrkni na [prodejny](https://myshop.cz.store-info.net/prodejny) hned.": "Mrkni na prodejny hned.",
    "Web najdeš na **[moncler.com](https://www.moncler.com/)** — sekce kolekcí.": "Web najdeš na **moncler.com** — sekce kolekcí.",
    "Přihlas se tady: https://myshop-cz.verify-login.com/account a hotovo.": "Přihlas se tady:  a hotovo.",
    "Nebo www.zalando.cz.": "Nebo .",
    "Viz <https://evil.example/x>, díky.": "Viz , díky.",
    "Napiš na [e-mail](mailto:info@myshop.cz).": "Napiš na [e-mail](mailto:info@myshop.cz).",
    "Napiš na [e-mail](mailto:x@evil.com).": "Napiš na e-mail.",
    "Podívej ![logo](https://evil.example/pixel.png) sem.": "Podívej !logo sem.",
    "Pole [1] a cena [3 000 Kč] nejsou odkazy.": "Pole [1] a cena [3 000 Kč] nejsou odkazy.",
    "Text bez odkazů, končí slovem": "Text bez odkazů, končí slovem",
}
OFFSITE = ["store-info.net", "moncler.com/", "verify-login", "zalando", "evil"]


@pytest.mark.parametrize(("text", "want"), CASES.items())
def test_sanitize(text, want):
    assert sanitize(text) == want


@pytest.mark.parametrize(("text", "want"), CASES.items())
def test_streaming_matches_sanitize_and_never_leaks(text, want):
    rng = random.Random(text)
    for _ in range(200):
        f, out, i = LinkFilter(), [], 0
        while i < len(text):
            n = rng.randint(1, 6)
            piece = f.feed(text[i : i + n])
            assert not any(h in piece for h in OFFSITE), piece
            out.append(piece)
            i += n
        out.append(f.flush())
        assert "".join(out) == want


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://myshop.cz/x", True),
        ("https://www.myshop.cz/x", True),
        ("https://blog.myshop.sk/a", True),
        ("/prodejny", True),
        ("tel:+420123456789", True),
        ("https://myshop.cz.store-info.net/p", False),
        ("https://evilmyshop.cz/p", False),
        ("https://myshop-cz.verify-login.com/", False),
    ],
)
def test_allowed(url, ok):
    assert allowed(url) is ok
