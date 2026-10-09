from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from assistant.runs.events import Event
from assistant.runs.links import sanitize
from assistant.utils.streaming import normalize_delta

CANCELLED_TOOL_RESULT = "Cancelled by the user before the tool finished."


def human_message_id(run_id: UUID) -> str:
    """Fixed per run, so a retried run never adds its input twice."""
    return f"run-{run_id}"


def ui_artifact(message: BaseMessage) -> dict | None:
    """The widget a tool attached to its result, if any."""
    artifact = getattr(message, "artifact", None)
    if isinstance(message, ToolMessage) and isinstance(artifact, dict):
        ui = artifact.get("ui")
        if isinstance(ui, dict):
            return ui
    return None


def to_ui_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Checkpointer messages as UI messages: each user message, then one
    assistant message holding the text and widgets of everything until the next one."""
    out: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    last_user: str | None = None

    def assistant(message_id: str | None) -> dict[str, Any]:
        nonlocal current
        if current is None:
            # Derived from the user message, so the UI's live message keeps its id when the run ends.
            current = {
                "id": f"{last_user}:answer" if last_user else message_id,
                "role": "assistant",
                "parts": [],
            }
            out.append(current)
        return current

    for m in messages:
        if isinstance(m, HumanMessage):
            current = None
            last_user = m.id
            out.append(
                {
                    "id": m.id,
                    "role": "user",
                    "parts": [{"type": "text", "text": normalize_delta(m.content)}],
                }
            )
        elif isinstance(m, AIMessage):
            # The checkpoint keeps the raw model text; the UI never sees off-site links.
            text = sanitize(normalize_delta(m.content))
            msg = assistant(m.id)
            if text:
                msg["parts"].append({"type": "text", "text": text})
        elif ui := ui_artifact(m):
            assistant(m.id)["parts"].append({"type": "ui", **ui})
    return out


def partial_texts(events: list[Event]) -> dict[str, str]:
    """Streamed text per model message since the last reset, in stream order."""
    texts: dict[str, str] = {}
    for e in events:
        if e.get("type") == "reset":
            texts = {}
        elif e.get("type") == "text-delta":
            texts[e["message_id"]] = texts.get(e["message_id"], "") + e["delta"]
    return texts


def cancel_updates(
    messages: list[BaseMessage], run_id: UUID, content: str, events: list[Event]
) -> list[BaseMessage]:
    """Messages that make a cancelled run's thread match what the user saw:
    the user input, results for unanswered tool calls, and the partial answer."""
    present = {m.id for m in messages}
    updates: list[BaseMessage] = []

    hid = human_message_id(run_id)
    if hid not in present:
        updates.append(HumanMessage(content=content, id=hid))

    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                if call["id"] not in answered:
                    updates.append(
                        ToolMessage(
                            content=CANCELLED_TOOL_RESULT, tool_call_id=call["id"]
                        )
                    )

    for message_id, text in partial_texts(events).items():
        if message_id not in present and text:
            updates.append(AIMessage(content=text, id=message_id))
    return updates
