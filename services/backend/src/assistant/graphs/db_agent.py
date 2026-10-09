from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelRequest,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    dynamic_prompt,
    wrap_model_call,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from assistant.config import settings
from assistant.models import chat_model

# Normal answers need up to ~10; past this the model gets an error per call and answers anyway.
MAX_TOOL_PER_RUN = 20

_prompt_cache: tuple[float, str] | None = None


def system_prompt() -> str:
    """The prompt file, re-read only when it changed, so edits apply without a restart."""
    global _prompt_cache
    path = Path(settings.system_prompt_file)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        raise RuntimeError(
            f"system prompt not found at {path}; create it or set SYSTEM_PROMPT_FILE"
        ) from None
    if _prompt_cache is None or _prompt_cache[0] != mtime:
        _prompt_cache = (mtime, path.read_text(encoding="utf-8"))
    return _prompt_cache[1]


@dynamic_prompt
def local_prompt(request: ModelRequest) -> str:
    return system_prompt()


IMAGE_PARTS = ("image_url", "image", "input_image")
OLD_IMAGE_NOTE = "[Product photo from earlier in the conversation, not sent again.]"


def without_images(message: ToolMessage) -> ToolMessage:
    if not isinstance(message.content, list):
        return message
    kept = [
        p
        for p in message.content
        if not (isinstance(p, dict) and p.get("type") in IMAGE_PARTS)
    ]
    if len(kept) == len(message.content):
        return message
    return message.model_copy(
        update={"content": [*kept, {"type": "text", "text": OLD_IMAGE_NOTE}]}
    )


@wrap_model_call
async def drop_old_images(request: ModelRequest, handler):
    """Images from earlier turns become a note in the model request (the checkpoint keeps them).

    The provider downloads every image in the history again on each call: a photo deleted
    from the shop would fail every later turn, and each image costs thousands of tokens.
    """
    messages = request.messages
    last_user = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1
    )
    trimmed = [
        without_images(m) if i < last_user and isinstance(m, ToolMessage) else m
        for i, m in enumerate(messages)
    ]
    return await handler(request.override(messages=trimmed))


def create_db_agent(
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    model_name: str | None = None,
):
    system_prompt()  # fails at startup if the prompt file is missing
    summary_model = chat_model(
        settings.summary_model_name,
        reasoning_effort="low",
        use_responses_api=True,
    )

    agent_model = chat_model(
        model_name or settings.model_name,
        streaming=True,
        temperature=0.1,
        # Includes reasoning; answers are short, so this only stops runaway output.
        max_tokens=settings.max_output_tokens,
        reasoning_effort="low",
        use_responses_api=True,
    )

    reserve = agent_model.max_tokens + 1000

    # No profile for OpenRouter names (openai/...); 100k is the conservative fallback.
    max_tokens_per_run = (agent_model.profile or {}).get(
        "max_input_tokens", 100_000
    ) - reserve

    middleware = [
        local_prompt,
        drop_old_images,
        ToolCallLimitMiddleware(
            thread_limit=10 * MAX_TOOL_PER_RUN,
            run_limit=MAX_TOOL_PER_RUN,
        ),
        # Cheap: clears old tool results.
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    keep=10,
                    trigger=int(0.8 * max_tokens_per_run),
                ),
            ],
        ),
        # Expensive fallback: summarizes the conversation.
        SummarizationMiddleware(
            model=summary_model,
            trigger=("tokens", max_tokens_per_run),
            keep=("messages", 20),
        ),
    ]

    return create_agent(
        agent_model,
        tools=tools,
        checkpointer=checkpointer,
        middleware=middleware,
    )
