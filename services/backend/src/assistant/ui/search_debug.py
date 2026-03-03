import streamlit as st
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import MatchAny

from assistant.api.config import settings

GROUPS = ["BOTY", "OBLEČENÍ", "BRÝLE", "DOPLŇKY", "VÝSTROJ", "OSTATNÍ"]
GENDERS = ["Dětské", "Dámské", "Pánské", "Uni"]

PAYLOAD_FIELDS = [
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
    "colors[].images",
]


@st.cache_resource
def get_client() -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key.get_secret_value(),
        grpc_port=settings.qdrant_grpc_port,
        https=False,
        prefer_grpc=True,
    )


@st.cache_resource
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-large")


def search(
    name: str,
    description: str,
    groups: list[str],
    genders: list[str],
    min_price: float | None,
    max_price: float | None,
    limit: int,
) -> list[dict]:
    client = get_client()
    emb = get_embeddings()

    name_emb = emb.embed_query(name)
    desc_emb = emb.embed_query(description)

    filters = [
        models.FieldCondition(
            key="colors[].price",
            range=models.Range(gte=min_price, lte=max_price),
        ),
    ]
    if groups:
        filters.append(models.FieldCondition(key="group", match=MatchAny(any=groups)))
    if genders:
        filters.append(models.FieldCondition(key="gender", match=MatchAny(any=genders)))

    global_filter = models.Filter(must=filters)

    res = client.query_points(
        collection_name="products",
        prefetch=[
            models.Prefetch(
                query=name_emb, using="name_emb", limit=limit, filter=global_filter
            ),
            models.Prefetch(
                query=desc_emb, using="desc_emb", limit=limit, filter=global_filter
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        query_filter=global_filter,
        limit=limit,
        with_payload=PAYLOAD_FIELDS,
    )

    products = []
    for pt in res.points:
        p = pt.payload or {}
        p["_score"] = pt.score
        products.append(p)
    return products


st.set_page_config(page_title="Product Search Debug", layout="wide")
st.title("Product Search Debug")

with st.sidebar:
    st.header("Query Parameters")
    name = st.text_input("Name (embedded)", value="zimní čepice pánská")
    description = st.text_input(
        "Description (embedded)", value="teplá pletená čepice pro muže"
    )
    groups = st.multiselect("Groups", GROUPS)
    genders = st.multiselect("Genders", GENDERS)

    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("Min price", min_value=0, value=0, step=100)
    with col2:
        max_price = st.number_input("Max price", min_value=0, value=0, step=100)

    limit = st.slider("Results limit", 1, 30, 10)

    run = st.button("Search", type="primary", use_container_width=True)

if run:
    with st.spinner("Searching..."):
        results = search(
            name=name,
            description=description,
            groups=groups if groups else [],
            genders=genders if genders else [],
            min_price=min_price if min_price > 0 else None,
            max_price=max_price if max_price > 0 else None,
            limit=limit,
        )

    st.subheader(f"{len(results)} results")

    for i, product in enumerate(results):
        colors = product.get("colors", [])
        score = product.get("_score", 0)
        gender = product.get("gender", "?")
        group = product.get("group", "?")
        subgroup = product.get("subgroup", "?")

        with st.expander(
            f"**{product.get('name', '?')}** — {gender} | {group}/{subgroup} | score: {score:.4f}",
            expanded=i < 3,
        ):
            st.caption(product.get("description_plain", "")[:300])

            if colors:
                cols = st.columns(min(len(colors), 4))
                for j, color in enumerate(colors):
                    with cols[j % len(cols)]:
                        images = color.get("images", [])
                        if images:
                            st.image(images[0], width=150)
                        st.markdown(f"**{color.get('color', '?')}**")
                        st.text(f"Code: {color.get('code', '?')}")
                        st.text(f"Price: {color.get('price', '?')} Kč")
                        url = color.get("url", "")
                        if url:
                            st.markdown(f"[Link]({url})")

            with st.popover("Raw JSON"):
                st.json(product)
