import pickle
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from api.config import settings

with open(Path(__file__).parent.parent.parent / "data" / "data.pickle", "rb") as f:
    points = pickle.load(f)["points"]

names = [x["payload"]["name"] for x in points]
descriptions = [x["payload"]["description_plain"] for x in points]

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:4b",
)


def embed_with_progress(emb, texts, desc="embedding"):
    """Embed texts in large batches and show a fast progress bar."""
    out = []
    for doc in tqdm(texts, desc=desc):
        out.extend(emb.embed_documents(doc))
    return out


names_emb = embed_with_progress(embeddings, names, "embedding names")
desc_emb = embed_with_progress(embeddings, descriptions, "embedding descriptions")

emb_length = len(desc_emb[0])

column_wise = {
    "ids": [x["id"] for x in points],
    "payloads": [x["payload"] for x in points],
    "vectors": {"name_emb": names_emb, "desc_emb": desc_emb},
}

client = QdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    api_key=settings.qdrant_api_key.get_secret_value(),
    grpc_port=settings.qdrant_grpc_port,
    https=False,
    prefer_grpc=True,
)

COLLECTION_NAME = "products"

client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "name_emb": models.VectorParams(
            size=emb_length, distance=models.Distance.COSINE
        ),
        "desc_emb": models.VectorParams(
            size=emb_length, distance=models.Distance.COSINE
        ),
    },
)

FIELD_SCHEMA = [
    ("name", "text"),
    ("group", "keyword"),
    ("subgroup", "keyword"),
    ("description_plain", "text"),
    ("colors[].price", "float"),
    ("colors[].color", "keyword"),
    ("colors[].sizes[].size", "keyword"),
]

for field, schema in FIELD_SCHEMA:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field,
        field_schema=schema,
    )

client.upsert(
    collection_name="products",
    points=models.Batch(**column_wise),
)
