"""
Product card widgets for ChatKit UI.
"""

from chatkit.widgets import (
    Card,
    Col,
    Image,
    ListView,
    ListViewItem,
    Row,
    Text,
    WidgetRoot,
)
from chatkit.actions import ActionConfig


def build_product_card(
    name: str,
    image_url: str | None,
    product_url: str,
    price: float | None = None,
) -> Card:
    """
    Build a single product card widget.
    
    Args:
        name: Product name
        image_url: URL to product image
        product_url: URL to product page (for click action)
        price: Optional product price
    
    Returns:
        Card widget with product information
    """
    children = []
    
    # Add image if available
    if image_url:
        children.append(
            Image(
                src=image_url,
                alt=name,
                height=120,
                width="100%",
                fit="contain",
                radius="md",
            )
        )
    
    # Add product name
    children.append(
        Text(
            value=name,
            weight="semibold",
            size="md",
            color="emphasis",
        )
    )
    
    # Add price if available
    if price is not None:
        children.append(
            Text(
                value=f"{price:.0f} Kč",
                size="sm",
                color="secondary",
            )
        )
    
    return Card(
        children=[
            Col(
                gap=2,
                children=children,
            )
        ],
        size="sm",
    )


def build_product_list_item(
    name: str,
    image_url: str | None,
    product_url: str,
    price: float | None = None,
) -> ListViewItem:
    """
    Build a single product list item for use in a ListView.
    
    Args:
        name: Product name
        image_url: URL to product image
        product_url: URL to product page (for click action)
        price: Optional product price
    
    Returns:
        ListViewItem widget with product information
    """
    children = []
    
    # Add image if available
    if image_url:
        children.append(
            Image(
                src=image_url,
                alt=name,
                size=60,
                fit="contain",
                radius="md",
                frame=True,
            )
        )
    
    # Add product details column
    detail_children = [
        Text(
            value=name,
            weight="medium",
            color="emphasis",
        )
    ]
    
    if price is not None:
        detail_children.append(
            Text(
                value=f"{price:.0f} Kč",
                size="sm",
                color="secondary",
            )
        )
    
    children.append(Col(children=detail_children))
    
    return ListViewItem(
        gap=3,
        children=children,
        onClickAction=ActionConfig(
            type="open_url",
            handler="client",
            payload={"url": product_url},
        ),
    )


def build_products_list(products: list[dict]) -> WidgetRoot:
    """
    Build a ListView widget containing multiple product cards.
    
    Args:
        products: List of product dictionaries with keys:
            - name: Product name
            - url: Product page URL
            - price: Product price
            - image: Product image URL
    
    Returns:
        ListView widget containing product items
    """
    items = []
    
    for product in products:
        name = product.get("name", "Unknown Product")
        image_url = product.get("image")
        product_url = product.get("url", "")
        price = product.get("price")
        
        items.append(
            build_product_list_item(
                name=name,
                image_url=image_url,
                product_url=product_url,
                price=price,
            )
        )
    
    return ListView(
        children=items,
        limit=5,
    )
