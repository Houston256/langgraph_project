from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph


def normalize_delta(c: list) -> str:
    out = ""
    for b in c:
        if isinstance(b, dict):
            out += b.get("text", "")
    return out

def create_config(thread_id: str, langfuse_handler) -> RunnableConfig:
    return RunnableConfig(
        configurable={"thread_id": thread_id},
        callbacks=[langfuse_handler],
        metadata={
            "langfuse_session_id": thread_id,
        },
    )

async def stream_graph_updates(
    user_input: str, graph: CompiledStateGraph, config, node_name: str = "model"
):
    chat_input = [HumanMessage(content=user_input)]

    async for chunk, metadata in graph.astream(
            MessagesState(messages=chat_input),
            stream_mode="messages",
            config=config
    ):
        if metadata.get("langgraph_node") != node_name:
            continue

        yield normalize_delta(getattr(chunk, "content", []))