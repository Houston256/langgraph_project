from typing import Annotated, Literal

import httpx
from langchain.tools import tool
from langchain_core.embeddings import Embeddings
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId
from langgraph.config import get_stream_writer
from langgraph.types import Command
from pydantic import BaseModel, Field, constr
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import MatchAny

from assistant.config import settings

COLLECTION = "products"

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
        default_factory=list,
        description='List of genders. Don\'t filter if empty. Prefer ["Pánské", "Uni"] for men, ["Dámské", "Uni"] for women.',
    )
    min_price: float | None = Field(
        None, description="Minimum price, don't filter if None"
    )
    max_price: float | None = Field(
        None, description="Maximum price, don't filter if None"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Exact terms results must contain in their name or description. Hard FILTER. "
            "Every entry must match; inside one entry, alternatives separated by | are OR-ed. "
            'Examples: ["3/4"], ["Gore-Tex", "merino"], ["tenis|tennis|raketa|racket"] to check '
            "whether the shop sells something at all. Case and diacritics insensitive, but no "
            "word forms: write the form a product name would contain, or list forms as "
            'alternatives ("bunda|bundy"). Leave empty for normal searches.'
        ),
    )


def create_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key.get_secret_value(),
        grpc_port=settings.qdrant_grpc_port,
        https=False,
        prefer_grpc=True,
    )


def product_to_card(product: dict, color: dict) -> dict:
    name = product.get("name", "Unknown Product")
    images = color.get("images") or []
    return {
        "name": name,
        "color": color.get("color"),
        "url": color.get("url", ""),
        "price": color.get("price"),
        "image": images[0] if images else None,
    }


async def image_exists(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5, follow_redirects=True) as http:
            r = await http.head(url)
        return r.status_code == 200 and r.headers.get("content-type", "").startswith(
            "image/"
        )
    except httpx.HTTPError:
        return False


def url_to_openai(url: str) -> dict[str, str]:
    return {
        "type": "image_url",
        "image_url": {"url": url},
    }


def create_product_tools(
    client: AsyncQdrantClient, embeddings: Embeddings
) -> list[BaseTool]:
    """Bind the catalog tools to a client and an embedding model."""

    @tool(parse_docstring=True)
    async def product_by_uuid(uuid: str) -> dict:
        """Find product by its unique identifier.

        Args:
            uuid: The unique identifier (slug) of the product to retrieve.

        Returns:
            Product payload if found, empty dict otherwise.
        """
        # Also finds products no longer in the feed, so old threads stay answerable.
        points, _ = await client.scroll(
            COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="slug", match=models.MatchValue(value=uuid)
                    )
                ]
            ),
            limit=1,
            with_payload=models.PayloadSelectorExclude(
                exclude=["emb_hash", "colors[].images"]
            ),
        )
        if not points:
            return {}
        payload = dict(points[0].payload or {})
        payload["uuid"] = payload.pop("slug")
        return payload

    @tool(
        args_schema=ProductFilterInput,
        description=(
            "Query a vector database of products. "
            "name and description are embedded for semantic RANKING only — they control result order, not filtering. "
            "Changing their wording won't fix empty results. "
            "groups, genders, min_price, max_price are hard FILTERS — if no results, loosen or remove them."
        ),
    )
    async def query_product(
        name: str,
        description: str,
        groups: list[cat_t],
        genders: list[gender_t],
        min_price: float | None = None,
        max_price: float | None = None,
        keywords: list[str] | None = None,
    ) -> list[dict]:
        """Query products by semantic search with optional filters."""
        writer = get_stream_writer()
        writer("Searching for products...")  # Progress message only

        groups_unique = list(set(groups))
        genders_unique = list(set(genders))

        name_emb, desc_emb = await embeddings.aembed_documents([name, description])

        filters = [
            # Out-of-stock products stay searchable; only ones dropped from the feed are skipped.
            models.FieldCondition(key="in_feed", match=models.MatchValue(value=True)),
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

        # Every entry must match; "a|b" matches a or b, as a phrase in the name or description.
        for entry in keywords or []:
            alternatives = [a.strip() for a in entry.split("|") if a.strip()]
            if alternatives:
                filters.append(
                    models.Filter(
                        should=[
                            models.FieldCondition(
                                key=field, match=models.MatchPhrase(phrase=alt)
                            )
                            for alt in alternatives
                            for field in ("name", "description_plain")
                        ]
                    )
                )

        global_filter = models.Filter(must=filters)

        res = await client.query_points(
            collection_name=COLLECTION,
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
                "colors[].in_stock",
                "colors[].sizes",
            ],
        )

        products = [x.payload for x in res.points]

        for p in products:
            p["uuid"] = p.pop("slug")

        return products

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
            A Command that posts a ToolMessage carrying the cards as its artifact.
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
            collection_name=COLLECTION,
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
                "colors[].color",
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
                "messages": [
                    ToolMessage(
                        content=f"Displayed {len(products)} product(s) as interactive cards.",
                        tool_call_id=tool_call_id,
                        artifact={
                            "ui": {"name": "products", "data": {"products": products}}
                        },
                    )
                ],
            }
        )

    @tool(parse_docstring=True)
    async def get_image(
        code: Annotated[
            str, "Color code from query_product results (colors[].code field)"
        ],
    ) -> list[dict[str, str]] | str:
        """Retrieve product image by color code for visual inspection.

        Args:
            code: The color code obtained from the colors[].code field of a previous query_product call. Identifies the specific product color variant.

        Returns:
            The product photo, "No photo available" if it has none, or an empty list for an unknown code.
        """
        writer = get_stream_writer()
        writer("Inspecting product images...")

        images = await client.query_points(
            collection_name=COLLECTION,
            limit=1,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="colors[].code",
                        match=models.MatchValue(value=code),
                    ),
                ]
            ),
            with_payload=["colors[].code", "colors[].images"],
        )
        if not images.points:
            return []
        colors = (images.points[0].payload or {}).get("colors") or []
        urls = next(
            (c.get("images") or [] for c in colors if c.get("code") == code), []
        )
        # The provider fails the whole request on an image it can't download.
        if not urls or not await image_exists(urls[0]):
            return "No photo available for this product."
        return [url_to_openai(urls[0])]

    return [product_by_uuid, query_product, display_products, get_image]
