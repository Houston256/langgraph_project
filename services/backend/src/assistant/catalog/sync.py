"""Syncs the product feed into Qdrant, once or daily.

python -m assistant.catalog.sync            # at start, then daily at catalog_sync_time
python -m assistant.catalog.sync --once     # one sync now
python -m assistant.catalog.sync --once --force --dry-run
"""

import argparse
import asyncio
import hashlib
import json
import logging
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
from langchain_core.embeddings import Embeddings
from psycopg.types.json import Jsonb
from qdrant_client import AsyncQdrantClient, models

from assistant.catalog.feed import Product, iter_products
from assistant.config import settings
from assistant.db import open_db
from assistant.models import embeddings as create_embeddings
from assistant.search.qdrant import COLLECTION, create_qdrant_client

log = logging.getLogger("catalog")

# Point ids are derived from the slug, so they stay the same across syncs.
ID_NAMESPACE = uuid.UUID("5d3c4a8e-6f0b-4b7e-9a51-1f2b8c0e7d42")
SYNC_LOCK_ID = 7_431_002
# Longer than any description in the feed (max ~3.4k chars), well under the model limit.
MAX_EMBED_CHARS = 8000
BATCH = 128

# Full text for keyword search: phrase matching keeps "3/4" together, folding makes "cepice" match "čepice".
TEXT_INDEX = models.TextIndexParams(
    type=models.TextIndexType.TEXT,
    tokenizer=models.TokenizerType.WORD,
    lowercase=True,
    ascii_folding=True,
    phrase_matching=True,
)

INDEXES = [
    ("slug", models.PayloadSchemaType.KEYWORD),
    ("name", TEXT_INDEX),
    ("description_plain", TEXT_INDEX),
    ("group", models.PayloadSchemaType.KEYWORD),
    ("subgroup", models.PayloadSchemaType.KEYWORD),
    ("gender", models.PayloadSchemaType.KEYWORD),
    ("in_feed", models.PayloadSchemaType.BOOL),
    ("colors[].code", models.PayloadSchemaType.KEYWORD),
    ("colors[].price", models.PayloadSchemaType.FLOAT),
    ("colors[].color", models.PayloadSchemaType.KEYWORD),
]


def point_id(slug: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, slug))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embed_texts(product: Product) -> tuple[str, str]:
    return product["name"], product["description_plain"][:MAX_EMBED_CHARS] or product[
        "name"
    ]


def embed_hash(product: Product) -> str:
    """Changes when the embedded text or the embedding model changes."""
    return _sha("\0".join([settings.embedding_model, *embed_texts(product)]))


def payload_hash(product: Product) -> str:
    return _sha(json.dumps(product, sort_keys=True, ensure_ascii=False))


Change = Literal["embedded", "updated", "unchanged"]


def change(stored: dict, product: Product) -> Change:
    """What the sync does with the product, from its stored hashes."""
    if stored.get("emb_hash") != product["emb_hash"]:
        return "embedded"
    changed = stored.get("payload_hash") != product["payload_hash"]
    # A product back in the feed has the same hash as before it left.
    returned = stored.get("in_feed") is False
    if changed or returned:
        return "updated"
    return "unchanged"


async def ensure_indexes(client: AsyncQdrantClient) -> None:
    """Creates missing payload indexes and replaces ones whose settings changed."""
    schema = (await client.get_collection(COLLECTION)).payload_schema
    for field, wanted in INDEXES:
        current = schema.get(field)
        if current is not None:
            if isinstance(wanted, models.TextIndexParams):
                if current.params == wanted:
                    continue
            elif current.data_type == wanted:
                continue
            log.info("replacing index on %s", field)
            await client.delete_payload_index(COLLECTION, field_name=field)
        await client.create_payload_index(
            COLLECTION, field_name=field, field_schema=wanted
        )


async def ensure_collection(client: AsyncQdrantClient, embeddings: Embeddings) -> None:
    if await client.collection_exists(COLLECTION):
        await ensure_indexes(client)
        return
    size = len(await embeddings.aembed_query("size probe"))
    log.info("creating collection %s (dim %s)", COLLECTION, size)
    await client.create_collection(
        COLLECTION,
        vectors_config={
            "name_emb": models.VectorParams(size=size, distance=models.Distance.COSINE),
            "desc_emb": models.VectorParams(size=size, distance=models.Distance.COSINE),
        },
    )
    await ensure_indexes(client)


async def stored_hashes(client: AsyncQdrantClient) -> dict[str, dict]:
    """id -> {emb_hash, payload_hash, in_feed} for every stored product."""
    if not await client.collection_exists(COLLECTION):
        return {}
    out: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = await client.scroll(
            COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=["emb_hash", "payload_hash", "in_feed"],
        )
        out.update({str(p.id): p.payload or {} for p in points})
        if offset is None:
            return out


class Writer:
    """Batches new and changed products into Qdrant writes."""

    def __init__(
        self, client: AsyncQdrantClient, embeddings: Embeddings, dry_run: bool
    ):
        self.client = client
        self.embeddings = embeddings
        self.dry_run = dry_run
        self.to_embed: list[tuple[str, Product]] = []
        self.to_update: list[tuple[str, Product]] = []

    async def embed(self, pid: str, product: Product) -> None:
        self.to_embed.append((pid, product))
        if len(self.to_embed) >= BATCH:
            await self.flush()

    async def update(self, pid: str, product: Product) -> None:
        self.to_update.append((pid, product))
        if len(self.to_update) >= BATCH:
            await self.flush()

    async def flush(self) -> None:
        embed, self.to_embed = self.to_embed, []
        update, self.to_update = self.to_update, []
        if self.dry_run:
            return
        if embed:
            texts = [embed_texts(p) for _, p in embed]
            names = await self.embeddings.aembed_documents([n for n, _ in texts])
            descs = await self.embeddings.aembed_documents([d for _, d in texts])
            await self.client.upsert(
                COLLECTION,
                points=[
                    models.PointStruct(
                        id=pid, vector={"name_emb": n, "desc_emb": d}, payload=p
                    )
                    for (pid, p), n, d in zip(embed, names, descs, strict=True)
                ],
            )
        if update:
            # Vectors stay; only the payload (prices, stock, ...) is replaced.
            await self.client.batch_update_points(
                COLLECTION,
                update_operations=[
                    models.OverwritePayloadOperation(
                        overwrite_payload=models.SetPayload(payload=p, points=[pid])
                    )
                    for pid, p in update
                ],
            )


async def sync(
    client: AsyncQdrantClient,
    embeddings: Embeddings,
    feed: AsyncIterator[bytes],
    dry_run: bool = False,
) -> dict[str, int]:
    if not dry_run:
        await ensure_collection(client, embeddings)
    stored = await stored_hashes(client)
    writer = Writer(client, embeddings, dry_run)
    stats: Counter[str] = Counter()
    seen: set[str] = set()

    async for product in iter_products(feed):
        pid = point_id(product["slug"])
        seen.add(pid)
        product = {**product, "in_feed": True}
        product["payload_hash"] = payload_hash(product)
        product["emb_hash"] = embed_hash(product)
        action = change(stored.get(pid, {}), product)
        stats[action] += 1
        if action == "embedded":
            await writer.embed(pid, product)
        elif action == "updated":
            await writer.update(pid, product)
    await writer.flush()

    # Kept so threads that showed them can still look them up; search skips them.
    gone = [
        pid for pid, p in stored.items() if pid not in seen and p.get("in_feed", True)
    ]
    stats["gone"] = len(gone)
    if gone and not dry_run:
        await client.set_payload(COLLECTION, payload={"in_feed": False}, points=gone)
    stats["products"] = len(seen)
    return dict(stats)


async def run_sync(force: bool = False, dry_run: bool = False) -> None:
    """One sync, unless another process holds the lock or the feed is unchanged."""
    async with open_db(max_size=2) as (pool, _), pool.connection() as conn:
        cur = await conn.execute(
            "SELECT pg_try_advisory_lock(%s) AS ok", (SYNC_LOCK_ID,)
        )
        if not (await cur.fetchone())["ok"]:  # type: ignore[index]
            log.info("another sync is running, skipping")
            return
        client = create_qdrant_client()
        try:
            empty = (
                not await client.collection_exists(COLLECTION)
                or (await client.count(COLLECTION)).count == 0
            )
            cur = await conn.execute(
                "SELECT last_modified FROM catalog_sync WHERE id = 1"
            )
            row = await cur.fetchone()
            since = None if force or empty or row is None else row["last_modified"]
            # Not the etag: this Apache adds "-gzip" to it and then never matches it again.
            headers = {"If-Modified-Since": since} if since else {}

            async with (
                httpx.AsyncClient(timeout=120, follow_redirects=True) as http,
                http.stream("GET", settings.catalog_feed_url, headers=headers) as r,
            ):
                if r.status_code == 304:
                    log.info("feed unchanged since %s", since)
                    return
                r.raise_for_status()
                stats = await sync(
                    client, create_embeddings(), r.aiter_bytes(), dry_run=dry_run
                )
                last_modified = r.headers.get("last-modified")
            log.info("sync %s %s", "plan" if dry_run else "done", stats)
            if not dry_run:
                await conn.execute(
                    "INSERT INTO catalog_sync (id, last_modified, synced_at, stats) VALUES (1, %s, now(), %s) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "last_modified = EXCLUDED.last_modified, synced_at = now(), stats = EXCLUDED.stats",
                    (last_modified, Jsonb(stats)),
                )
        finally:
            await client.close()
            await conn.execute("SELECT pg_advisory_unlock(%s)", (SYNC_LOCK_ID,))


def seconds_until(at: str, tz: str) -> float:
    """Seconds until the next HH:MM in the time zone."""
    now = datetime.now(ZoneInfo(tz))
    hour, minute = map(int, at.split(":"))
    nxt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if nxt <= now:
        nxt += timedelta(days=1)
    return (nxt - now).total_seconds()


async def daily() -> None:
    while True:
        # The run at start fills an empty collection; otherwise it's a no-op until the feed changes.
        try:
            await run_sync()
        except Exception:
            log.exception("sync failed")
        wait = seconds_until(settings.catalog_sync_time, settings.catalog_sync_tz)
        log.info("next sync in %.0f s", wait)
        await asyncio.sleep(wait)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="sync now and exit")
    parser.add_argument(
        "--force", action="store_true", help="sync even if the feed is unchanged"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan, write nothing"
    )
    args = parser.parse_args()
    if args.once:
        asyncio.run(run_sync(force=args.force, dry_run=args.dry_run))
    else:
        asyncio.run(daily())


if __name__ == "__main__":
    main()
