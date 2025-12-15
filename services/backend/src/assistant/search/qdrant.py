from typing import Annotated, Literal, Optional

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langchain_ollama import OllamaEmbeddings
from langgraph.config import get_stream_writer
from langgraph.types import Command
from pydantic import BaseModel, Field, constr
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import MatchAny

from assistant.api.config import settings

client = AsyncQdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    api_key=settings.qdrant_api_key.get_secret_value(),
    grpc_port=settings.qdrant_grpc_port,
    https=False,
    prefer_grpc=True,
)

embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

cat_t = Literal["BOTY", "OBLEČENÍ", "BRÝLE", "DOPLŇKY", "VÝSTROJ", "OSTATNÍ"]
gender_t = Literal["Dětské", "Dámské", "Pánské", "Uni"]

NonEmptyStr = constr(min_length=1)


class ProductFilterInput(BaseModel):
    name: NonEmptyStr = Field(
        description="Product name, plain text which will be embedded"
    )
    description: NonEmptyStr = Field(
        description="Product description, plain text which will be embedded"
    )
    groups: list[cat_t] = Field(
        default_factory=list, description="List of groups. Don't filter if empty"
    )
    genders: list[gender_t] = Field(
        default_factory=list, description="List of genders. Don't filter if empty"
    )
    min_price: Optional[float] = Field(None, description="Minimum price")
    max_price: Optional[float] = Field(None, description="Maximum price")


@tool(parse_docstring=True)
async def product_by_uuid(uuid: str) -> dict:
    """Find product by its unique identifier.

    Args:
        uuid: The unique identifier (slug) of the product to retrieve.

    Returns:
        Product payload if found, empty dict otherwise.
    """
    point = await client.query_points(collection_name="products", query=uuid)
    if not point.points:
        return {}
    return point.points[0].payload


@tool(
    args_schema=ProductFilterInput,
    description="Query a vector database of products to find matching items based on semantic search and filters.",
)
async def query_product(
    name: str,
    description: str,
    groups: list[cat_t],
    genders: list[gender_t],
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> list[dict]:
    """Query products by semantic search with optional filters."""
    writer = get_stream_writer()
    writer("Searching for products...")  # Progress message only

    groups_unique = list(set(groups))
    genders_unique = list(set(genders))

    name_emb = await embeddings.aembed_query(name)
    desc_emb = await embeddings.aembed_query(description)

    filters = [
        models.FieldCondition(
            key="colors[].price",
            range=models.Range(
                gte=min_price,
                lte=max_price,
            ),
        ),
    ]
    if groups:
        filters.append(
            models.FieldCondition(key="group", match=MatchAny(any=groups_unique))
        )

    if genders:
        filters.append(
            models.FieldCondition(key="gender", match=MatchAny(any=genders_unique))
        )

    global_filter = models.Filter(must=filters)

    res = await client.query_points(
        collection_name="products",
        prefetch=[
            models.Prefetch(
                query=name_emb,
                using="name_emb",
                limit=10,
                filter=global_filter,
            ),
            models.Prefetch(
                query=desc_emb,
                using="desc_emb",
                limit=10,
                filter=global_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        query_filter=global_filter,
        limit=10,
        with_payload=[
            "slug",
            "name",
            "description_plain",
            "group",
            "subgroup",
            "gender",
            "colors[].code",
            "colors[].color",
            "colors[].url",
            "colors[].price",
        ],
    )

    products = [x.payload for x in res.points]

    for p in products:
        p["uuid"] = p.pop("slug")

    return products


def product_to_card(product: dict, color: dict) -> dict:
    name = product.get("name", "Unknown Product")
    images = color.get("images") or []
    return {
        "name": name,
        "url": color.get("url", ""),
        "price": color.get("price"),
        "image": images[0] if images else None,
    }


@tool(parse_docstring=True)
async def display_products(
    color_codes: Annotated[
        list[str],
        "List of color codes from query_product results (colors[].code field)",
    ],
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Display products as interactive cards.

    Args:
        color_codes: Color codes from `query_product` results (the `colors[].code` field).
        tool_call_id: Injected tool call id (provided by LangChain).

    Returns:
        A Command that updates the UI widget and posts a ToolMessage.
    """
    writer = get_stream_writer()
    writer("Loading product details...")

    codes = list(dict.fromkeys([c for c in color_codes if c]))
    if not codes:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No products to display.", tool_call_id=tool_call_id
                    )
                ],
            }
        )

    res = await client.query_points(
        collection_name="products",
        query=None,
        limit=len(codes),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="colors[].code",
                    match=models.MatchAny(any=codes),
                )
            ]
        ),
        with_payload=[
            "name",
            "colors[].code",
            "colors[].url",
            "colors[].price",
            "colors[].images",
        ],
    )

    wanted = set(codes)
    code_to_card: dict[str, dict] = {}

    for pt in res.points:
        payload = pt.payload or {}
        colors = payload.get("colors") or []
        for c in colors:
            code = c.get("code")
            if code in wanted and code not in code_to_card:
                code_to_card[code] = product_to_card(payload, c)

    products = [code_to_card[c] for c in codes if c in code_to_card]
    if not products:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Could not find any products with the provided color codes.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            "widget": {"type": "products_widget", "products": products},
            "messages": [
                ToolMessage(
                    content=f"Displayed {len(products)} product(s) as interactive cards.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def url_to_openai(url: str) -> dict[str, str]:
    return {
        "type": "image_url",
        "image_url": {"url": url},
    }


@tool(parse_docstring=True)
async def get_image(
    code: Annotated[str, "Color code from query_product results (colors[].code field)"],
) -> list[dict[str, str]]:
    """Retrieve product image by color code for visual inspection.

    Args:
        code: The color code obtained from the colors[].code field of a previous query_product call. Identifies the specific product color variant.

    Returns:
        List containing image object formatted for OpenAI vision API, or empty list if no image found.
    """
    writer = get_stream_writer()
    writer("Inspecting product images...")

    images = await client.query_points(
        collection_name="products",
        limit=1,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="colors[].code",
                    match=models.MatchValue(value=code),
                ),
            ]
        ),
        with_payload=["colors[].images"],
    )
    try:
        img_url = images.points[0].payload["colors"][0]["images"][0]
    except IndexError:
        return []
    return [url_to_openai(img_url)]
