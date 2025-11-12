from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ContextEditingMiddleware,
    ClearToolUsesEdit,
)
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from api.config import settings
from graphs.prompts import SYSTEM_PROMPT
from search.qdrant import query_product, get_image

MAX_TOOL_PER_RUN = 10

summary_model = init_chat_model(
    "gpt-5-nano",
    reasoning_effort="minimal",
    use_responses_api=True,
)

agent_model = init_chat_model(
    settings.model_name,
    streaming=True,
    temperature=0.4,
    max_tokens=5000,
    timeout=30,
    reasoning_effort="minimal",
    use_responses_api=True,
)

reserve = agent_model.max_tokens + 1000

MAX_TOKENS_PER_RUN = agent_model.profile.get("max_input_tokens", 100_000) - reserve

middleware = [
    ToolCallLimitMiddleware(
        thread_limit=10 * MAX_TOOL_PER_RUN,
        run_limit=MAX_TOOL_PER_RUN,
    ),
    # (fast method) minimize context by clearing tool calls
    ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                keep=10,
                trigger=int(0.8 * MAX_TOKENS_PER_RUN),
            ),
        ],
    ),
    # (slow, expensive fallback) summarize convo
    SummarizationMiddleware(
        model=summary_model,
        max_tokens_before_summary=MAX_TOKENS_PER_RUN,
        messages_to_keep=20,
    ),
]


def create_db_agent():
    return create_agent(
        agent_model,
        tools=[query_product, get_image],
        checkpointer=InMemorySaver(),
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
    )
