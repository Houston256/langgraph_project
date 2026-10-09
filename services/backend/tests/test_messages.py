from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from assistant.runs.messages import (
    CANCELLED_TOOL_RESULT,
    cancel_updates,
    human_message_id,
    partial_texts,
    to_ui_messages,
)

CARDS = {"ui": {"name": "products", "data": {"products": [{"name": "Mikina"}]}}}


def test_ui_messages_group_a_turn_into_one_answer_with_a_stable_id():
    messages = [
        HumanMessage("Ukaž mikiny", id="run-1"),
        AIMessage(
            "",
            id="resp_a",
            tool_calls=[{"name": "display_products", "args": {}, "id": "c1"}],
        ),
        ToolMessage(
            "Displayed 1 product(s).", tool_call_id="c1", artifact=CARDS, id="t1"
        ),
        AIMessage("Tady jsou. Víc na [webu](https://evil.example).", id="resp_b"),
    ]
    user, answer = to_ui_messages(messages)
    assert user == {
        "id": "run-1",
        "role": "user",
        "parts": [{"type": "text", "text": "Ukaž mikiny"}],
    }
    # Same id the frontend gives the live answer, so the UI doesn't show a second branch.
    assert answer["id"] == "run-1:answer"
    assert answer["parts"] == [
        {"type": "ui", "name": "products", "data": {"products": [{"name": "Mikina"}]}},
        {"type": "text", "text": "Tady jsou. Víc na webu."},
    ]


def test_partial_texts_start_over_at_reset():
    events = [
        {"type": "text-delta", "message_id": "m1", "delta": "Ahoj "},
        {"type": "reset"},
        {"type": "text-delta", "message_id": "m2", "delta": "Znovu"},
    ]
    assert partial_texts(events) == {"m2": "Znovu"}


def test_cancel_adds_input_tool_results_and_partial_answer():
    run_id = uuid4()
    hid = human_message_id(run_id)
    messages = [
        HumanMessage("Hledám bundu", id=hid),
        AIMessage(
            "",
            id="resp_a",
            tool_calls=[{"name": "query_product", "args": {}, "id": "c1"}],
        ),
    ]
    events = [{"type": "text-delta", "message_id": "resp_b", "delta": "Vybrala jsem"}]
    updates = cancel_updates(messages, run_id, "Hledám bundu", events)
    assert [type(m).__name__ for m in updates] == ["ToolMessage", "AIMessage"]
    assert (
        updates[0].tool_call_id == "c1" and updates[0].content == CANCELLED_TOOL_RESULT
    )
    assert updates[1].content == "Vybrala jsem"


def test_cancel_before_start_keeps_the_message():
    run_id = uuid4()
    (update,) = cancel_updates([], run_id, "Ahoj", [])
    assert isinstance(update, HumanMessage) and update.id == human_message_id(run_id)
