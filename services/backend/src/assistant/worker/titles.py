import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage

from assistant.config import settings
from assistant.db import Pool
from assistant.models import chat_model
from assistant.runs import store
from assistant.utils.streaming import normalize_delta

log = logging.getLogger(__name__)

TITLE_PROMPT = (
    "Name the topic of a shop chat that starts with the user's message below, in 2 to 5 words. "
    "Don't copy the message and don't add anything it doesn't say. Use the message's language "
    "with correct spelling and diacritics. Reply with the title only, no quotes, no final period."
)


async def make_title(
    pool: Pool, run: dict[str, Any], callbacks: list[BaseCallbackHandler]
) -> None:
    """Names the thread after its first message; failures only leave it untitled."""
    try:
        llm = chat_model(
            settings.model_name, reasoning_effort="low", use_responses_api=True
        )
        reply = await llm.ainvoke(
            [SystemMessage(TITLE_PROMPT), HumanMessage(run["input"]["content"][:1000])],
            config={"callbacks": callbacks},
        )
        title = normalize_delta(reply.content).strip().strip('"').rstrip(".")[:80]
        if title:
            await store.set_title(pool, run["thread_id"], title)
    except Exception:
        log.exception("title for thread %s failed", run["thread_id"])
