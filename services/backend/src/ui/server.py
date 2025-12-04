from datetime import datetime, timezone, timedelta
from typing import AsyncIterator
from langfuse.langchain import CallbackHandler

from chatkit.server import ChatKitServer
from chatkit.types import (
    ThreadMetadata,
    UserMessageItem,
    AssistantMessageItem,
    ThreadItemAddedEvent,
    ThreadItemUpdated,
    ThreadItemDoneEvent,
    AssistantMessageContentPartTextDelta,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartAdded,
    ThreadStreamEvent,
    AssistantMessageContent,
)
from graphs.db_agent import create_db_agent
from utils.streaming import stream_graph_updates, create_config


def to_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class LangGraphChatKitServer(ChatKitServer[dict]):
    def __init__(self, store):
        super().__init__(store)
        self.graph = create_db_agent()
        self.langfuse_handler = CallbackHandler()

    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: dict,
    ) -> AsyncIterator[ThreadStreamEvent]:
        thread_items_page = await self.store.load_thread_items(
            thread.id,
            after=None,
            limit=100,
            order="asc",
            context=context,
        )

        messages_for_graph: list[dict[str, str]] = []
        seen_ids = {it.id for it in thread_items_page.data}

        for item in thread_items_page.data:
            if isinstance(item, UserMessageItem):
                for part in item.content:
                    if part.type == "input_text":
                        messages_for_graph.append(
                            {"role": "user", "content": part.text}
                        )
            elif isinstance(item, AssistantMessageItem):
                buf: list[str] = []
                for part in item.content:
                    if part.type == "output_text":
                        buf.append(part.text)
                if buf:
                    messages_for_graph.append(
                        {"role": "assistant", "content": "".join(buf)}
                    )

        if input_user_message is not None and input_user_message.id not in seen_ids:
            for part in input_user_message.content:
                if part.type == "input_text":
                    messages_for_graph.append({"role": "user", "content": part.text})

        item_times = [to_aware_utc(it.created_at) for it in thread_items_page.data]
        last_ts = (
            max(item_times) if item_times else datetime.min.replace(tzinfo=timezone.utc)
        )
        assistant_created_at = max(
            datetime.now(timezone.utc), last_ts + timedelta(microseconds=1)
        )

        assistant_item_id = self.store.generate_item_id("message", thread, context)
        assistant_item = AssistantMessageItem(
            id=assistant_item_id,
            thread_id=thread.id,
            created_at=assistant_created_at,
            content=[],
            type="assistant_message",
        )

        yield ThreadItemAddedEvent(item=assistant_item)

        yield ThreadItemUpdated(
            item_id=assistant_item_id,
            update=AssistantMessageContentPartAdded(
                type="assistant_message.content_part.added",
                content_index=0,
                content=AssistantMessageContent(type="output_text", text=""),
            ),
        )

        full_text: list[str] = []

        config = create_config(thread_id=thread.id, langfuse_handler=self.langfuse_handler)

        last = messages_for_graph[-1].get("content", "")

        async for delta in stream_graph_updates(last, self.graph, config):
            if not delta:
                continue
            full_text.append(delta)
            yield ThreadItemUpdated(
                item_id=assistant_item_id,
                update=AssistantMessageContentPartTextDelta(
                    type="assistant_message.content_part.text_delta",
                    content_index=0,
                    delta=delta,
                ),
            )

        final_text = "".join(full_text)
        content = AssistantMessageContent(type="output_text", text=final_text)

        yield ThreadItemUpdated(
            item_id=assistant_item_id,
            update=AssistantMessageContentPartDone(
                type="assistant_message.content_part.done",
                content_index=0,
                content=content,
            ),
        )

        final_assistant_item = AssistantMessageItem(
            id=assistant_item_id,
            thread_id=thread.id,
            created_at=assistant_item.created_at,
            type="assistant_message",
            content=[content],
        )

        yield ThreadItemDoneEvent(item=final_assistant_item)
