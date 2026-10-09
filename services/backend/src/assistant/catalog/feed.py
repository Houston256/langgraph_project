"""Google Merchant feed → product payloads for Qdrant, parsed as a stream."""

import re
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

G = "{http://base.google.com/ns/1.0}"

# First level of g:product_type, last word ("Cloudmonster 1 Muži" -> "Muži").
GENDERS = {"Muži": "Pánské", "Ženy": "Dámské", "Děti": "Dětské", "Uni": "Uni"}

# ".../damska-mikina-varley--1005848.html" -> slug, color code
LINK_RE = re.compile(r"/([^/]+)--(\d+)\.html")

Product = dict[str, Any]


class FeedError(Exception):
    pass


def _text(item: ET.Element, tag: str) -> str:
    return (item.findtext(tag) or "").strip()


def _price(value: str) -> float | None:
    # "3050.00 CZK"
    try:
        return float(value.split()[0])
    except IndexError, ValueError:
        return None


def _categories(product_type: str) -> tuple[str, str | None, str | None]:
    """'Ženy - Oblečení - Mikiny' -> ('Dámské', 'OBLEČENÍ', 'MIKINY')."""
    parts = [p.strip() for p in product_type.split(" - ")]
    gender = GENDERS.get(parts[0].split()[-1] if parts[0] else "", "Uni")
    group = parts[1].upper() if len(parts) > 1 else None
    subgroup = " - ".join(parts[2:]).upper() if len(parts) > 2 else None
    return gender, group, subgroup


def _clean_url(url: str) -> str:
    """Product link without utm query and fragment."""
    s = urlsplit(url)
    return urlunsplit((s.scheme, s.netloc, s.path, "", ""))


def _new_product(slug: str, item: ET.Element) -> Product:
    gender, group, subgroup = _categories(_text(item, G + "product_type"))
    return {
        "slug": slug,
        "name": _text(item, "title"),
        "description_plain": _text(item, G + "description"),
        "brand": _text(item, G + "brand"),
        "gender": gender,
        "group": group,
        "subgroup": subgroup,
        "colors": {},
    }


def _add_item(product: Product, code: str, link: str, item: ET.Element) -> None:
    """Adds one feed item (one size of one color) to its product."""
    color = product["colors"].get(code)
    if color is None:
        regular = _price(_text(item, G + "price"))
        sale = _price(_text(item, G + "sale_price"))
        images = [_text(item, G + "image_link")] + [
            (e.text or "").strip() for e in item.findall(G + "additional_image_link")
        ]
        color = product["colors"][code] = {
            "code": code,
            "color": _text(item, G + "color"),
            "url": _clean_url(link),
            "price": sale if sale is not None else regular,
            "regular_price": regular,
            "images": [i for i in images if i],
            "sizes": [],
        }
    color["sizes"].append(
        {
            "size": _text(item, G + "size"),
            "in_stock": _text(item, G + "availability") == "in stock",
        }
    )


def _finish(product: Product) -> Product:
    product["colors"] = list(product["colors"].values())
    for color in product["colors"]:
        color["in_stock"] = any(s["in_stock"] for s in color["sizes"])
    product["in_stock"] = any(c["in_stock"] for c in product["colors"])
    return product


async def iter_products(chunks: AsyncIterator[bytes]) -> AsyncIterator[Product]:
    """Products from the raw feed, one at a time.

    Holds only the current product; relies on the feed listing a product's items together.
    """
    parser = ET.XMLPullParser(events=("start", "end"))
    channel: ET.Element | None = None
    current: Product | None = None
    done: set[str] = set()

    def items() -> list[ET.Element]:
        nonlocal channel
        out = []
        for event, elem in parser.read_events():
            if event == "start" and elem.tag == "channel":
                channel = elem
            elif event == "end" and elem.tag == "item":
                out.append(elem)
        return out

    async def chunks_then_end() -> AsyncIterator[bytes | None]:
        async for chunk in chunks:
            yield chunk
        yield None

    async for chunk in chunks_then_end():
        if chunk is None:
            parser.close()
        else:
            parser.feed(chunk)
        for item in items():
            link = _text(item, "link")
            m = LINK_RE.search(link)
            if m:
                slug, code = m.group(1), _text(item, G + "item_group_id") or m.group(2)
                if current is not None and current["slug"] != slug:
                    done.add(current["slug"])
                    yield _finish(current)
                    current = None
                if current is None:
                    if slug in done:
                        raise FeedError(f"items of {slug!r} are not listed together")
                    current = _new_product(slug, item)
                _add_item(current, code, link, item)
            # Parsed items would otherwise pile up under <channel>.
            if channel is not None:
                channel.remove(item)
    if current is not None:
        yield _finish(current)
