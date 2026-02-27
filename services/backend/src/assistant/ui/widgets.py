from chatkit.actions import ActionConfig
from chatkit.widgets import (
    Box,
    Button,
    Card,
    Col,
    Image,
    Text,
    DynamicWidgetRoot,
)

def build_product_tile(name: str, image_url: str | None, product_url: str, price: float | None = None) -> Col:
    children = []
    if image_url:
        children.append(Image(src=image_url, alt=name, height=120, width="100%", fit="contain", radius="md"))
    children.append(Text(value=name, weight="semibold", maxLines=2, color="emphasis"))
    if price is not None:
        children.append(Text(value=f"{price:.0f} Kč", size="sm", color="secondary"))
    children.append(
        Button(
            label="Open",
            variant="soft",
            onClickAction=ActionConfig(type="open_url", handler="client", payload={"url": product_url}),
        )
    )
    return Col(
        width=240,
        minWidth=240,
        flex="0 0 auto",
        padding=3,
        gap=2,
        radius="lg",
        border=1,
        children=children,
    )

def build_products_carousel(products: list[dict]) -> DynamicWidgetRoot:
    tiles = [
        build_product_tile(
            name=p.get("name", "Unknown Product"),
            image_url=p.get("image"),
            product_url=p.get("url", ""),
            price=p.get("price"),
        )
        for p in products[:10]
    ]

    widget = Card(
        size="full",
        padding=0,
        children=[
            Box(
                direction="row",
                wrap="nowrap",
                gap=3,
                padding={"x": 3, "y": 2},
                # consider removing width="100%" if you’re experimenting with overflow behavior
                children=tiles,
            )
        ],
    )

    d = widget.model_dump(exclude_none=True)
    d["children"][0]["scrollable"] = True  # inject extra prop on the Box
    return DynamicWidgetRoot.model_validate(d)
