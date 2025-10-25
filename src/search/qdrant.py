from typing import Optional, Literal

from langchain.tools import tool
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
from qdrant_client import models, AsyncQdrantClient
from qdrant_client.http.models import MatchAny

from api.config import settings

client = AsyncQdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    api_key=settings.qdrant_api_key.get_secret_value(),
    grpc_port=settings.qdrant_grpc_port,
    https=False,
    prefer_grpc=True
)

embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

cat_t = Literal['BOTY', 'OBLEČENÍ', 'BRÝLE', 'DOPLŇKY', 'VÝSTROJ', 'OSTATNÍ']
gender_t = Literal['Dětské', 'Dámské', 'Pánské', "Uni"]


class ProductFilterInput(BaseModel):
    name: str = Field(description="Product name, plain text which will be embedded")
    description: str = Field(description="Product description, plain text which will be embedded")
    groups: set[cat_t] = Field(default_factory=set, description="Set of groups. Don't filter if empty")
    genders: set[gender_t] = Field(default_factory=set, description="Set of genders. Don't filter if empty")
    min_price: Optional[float] = Field(None, description="Minimum price")
    max_price: Optional[float] = Field(None, description="Maximum price")


@tool
async def product_by_uuid(uuid: str) -> dict:
    """
    Find product by its uuid.
    :param uuid: unique id of product
    :return: product if exists, empty dict otherwise
    """
    point = await client.query_points(
        collection_name="products",
        query=uuid
    )
    if not point.points:
        return {}
    return point.points[0].payload


@tool(args_schema=ProductFilterInput, description="Query a vector database of products.")
async def query_product(name: str,
                        description: str,
                        groups: set[cat_t] = set(),
                        genders: set[gender_t] = set(),
                        min_price: Optional[float] = None,
                        max_price: Optional[float] = None,
                        ) -> list[dict]:
    """
    Query a vector database of products.
    :param name: expected name of product
    :param description: expected description of product
    :param groups: set of groups to filter by, dont filter if empty
    :param genders: set of genders to filter by, dont filter if empty
    :param min_price: minimal price of product
    :param max_price: maximal price of product
    :return: list of products which satisfy the query
    """

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
            models.FieldCondition(
                key="group",
                match=MatchAny(any=list(groups))
            ))

    if genders:
        filters.append(
            models.FieldCondition(
                key="gender",
                match=MatchAny(any=list(genders))
            ))

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
        with_payload=["slug", "name", "description_plain", "group", "subgroup", "gender",
                      "colors[].color", "colors[].url", "colors[].price", ]
    )

    res = [x.payload for x in res.points]

    for p in res:
        p['uuid'] = p.pop('slug')

    return res
