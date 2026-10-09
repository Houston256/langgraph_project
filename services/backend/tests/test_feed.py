import pytest

from assistant.catalog.feed import FeedError, iter_products


def item(
    code: str, slug: str, size: str, stock: str = "in stock", sale: str = ""
) -> str:
    sale_tag = f"<g:sale_price>{sale} CZK</g:sale_price>" if sale else ""
    return f"""
<item>
  <g:id>{code}-{size}</g:id>
  <title>Dámská mikina Varley Always</title>
  <g:description>Mikina na zip.</g:description>
  <link>https://www.myshop.cz/{slug}--{code}.html?utm_source=google#x</link>
  <g:image_link>https://www.myshop.cz/WINSHOP/foto_temp/{code}_1.png</g:image_link>
  <g:additional_image_link>https://www.myshop.cz/WINSHOP/foto_temp/{code}_2.png</g:additional_image_link>
  <g:availability>{stock}</g:availability>
  <g:price>3050.00 CZK</g:price>{sale_tag}
  <g:product_type> Cloudmonster 1 Ženy - Oblečení - Mikiny</g:product_type>
  <g:brand>Varley</g:brand>
  <g:item_group_id>{code}</g:item_group_id>
  <g:color>Snow white</g:color>
  <g:gender>male</g:gender>
  <g:size>{size}</g:size>
</item>"""


def feed(*items: str) -> bytes:
    body = "".join(items)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:g="http://base.google.com/ns/1.0"><channel>{body}</channel></rss>""".encode()


async def chunks(data: bytes, size: int = 37):
    for i in range(0, len(data), size):
        yield data[i : i + size]


async def parse(data: bytes) -> list[dict]:
    return [p async for p in iter_products(chunks(data))]


async def test_groups_sizes_into_colors_and_colors_into_products():
    data = feed(
        item("1005848", "damska-mikina", "M"),
        item("1005848", "damska-mikina", "L", stock="out of stock"),
        item("1005847", "damska-mikina", "S", sale="2135.00"),
        item("2002198", "panska-bunda", "XL"),
    )
    products = await parse(data)
    assert [p["slug"] for p in products] == ["damska-mikina", "panska-bunda"]
    mikina = products[0]
    # g:gender says "male" for everything; product_type is right.
    assert (mikina["gender"], mikina["group"], mikina["subgroup"]) == (
        "Dámské",
        "OBLEČENÍ",
        "MIKINY",
    )
    white, sale = mikina["colors"]
    assert white["url"] == "https://www.myshop.cz/damska-mikina--1005848.html"
    assert [s["size"] for s in white["sizes"]] == ["M", "L"]
    assert white["in_stock"] is True and white["sizes"][1]["in_stock"] is False
    assert (sale["price"], sale["regular_price"]) == (2135.0, 3050.0)
    assert len(white["images"]) == 2


async def test_product_listed_twice_aborts():
    data = feed(
        item("1", "a", "M"),
        item("2", "b", "M"),
        item("1", "a", "L"),
    )
    with pytest.raises(FeedError):
        await parse(data)
