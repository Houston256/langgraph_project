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
    user_input: str,
    graph: CompiledStateGraph,
    config,
    node_name: str = "model",
    custom: bool = False,
):
    chat_input = [HumanMessage(content=user_input)]
    stream_modes = ["messages"] + (["updates", "custom"] if custom else [])

    async for mode, payload in graph.astream(
        MessagesState(messages=chat_input),  # type: ignore
        stream_mode=stream_modes,  # type: ignore
        config=config,
    ):
        if mode == "messages":
            chunk, metadata = payload
            if metadata.get("langgraph_node") != node_name:
                continue
            content = normalize_delta(getattr(chunk, "content", []))
            yield (mode, content) if custom else content
        elif mode == "updates":
            if isinstance(payload, dict):
                for node_updates in payload.values():
                    if isinstance(node_updates, dict):
                        widget = node_updates.get("widget")
                        if widget is not None:
                            yield ("widget", widget)
        elif custom:
            yield mode, payload
