from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from assistant.graphs.db_agent import create_db_agent
from assistant.models import embeddings
from assistant.search.qdrant import (
    create_product_tools,
    create_qdrant_client,
)

DEFAULT_GRAPH = "agent"


@dataclass(frozen=True)
class GraphSpec:
    graph: CompiledStateGraph
    # Node whose model output is the answer text shown to the user.
    text_node: str


@asynccontextmanager
async def open_graphs(
    checkpointer: BaseCheckpointSaver,
) -> AsyncIterator[dict[str, GraphSpec]]:
    qdrant = create_qdrant_client()
    try:
        tools = create_product_tools(qdrant, embeddings())
        yield {
            "agent": GraphSpec(create_db_agent(tools, checkpointer), "model"),
        }
    finally:
        await qdrant.close()
