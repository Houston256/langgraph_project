from langchain_core.runnables import RunnableConfig


def normalize_delta(c: str | list) -> str:
    # Plain chat models stream str deltas, the responses API streams content blocks.
    if isinstance(c, str):
        return c
    out = ""
    for b in c:
        if isinstance(b, dict):
            out += b.get("text", "")
    return out


def create_config(thread_id: str) -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": thread_id})
