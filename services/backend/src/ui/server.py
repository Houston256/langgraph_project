from datetime import datetime, timezone, timedelta
from typing import AsyncIterator, Iterable

from langfuse.langchain import CallbackHandler

from chatkit.server import ChatKitServer
from chatkit.types import (
    ThreadMetadata,
    UserMessageItem,
    AssistantMessageItem,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    AssistantMessageContentPartTextDelta,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartAdded,
    ThreadStreamEvent,
    AssistantMessageContent,
    ProgressUpdateEvent,
    ThreadItemUpdatedEvent,
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

    @staticmethod
    def _extract_text_messages(items: Iterable[object]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        for item in items:
            if isinstance(item, UserMessageItem):
                for part in item.content:
                    if part.type == "input_text":
                        messages.append({"role": "user", "content": part.text})
                continue

            if isinstance(item, AssistantMessageItem):
                buf: list[str] = []
                for part in item.content:
                    if part.type == "output_text":
                        buf.append(part.text)
                if buf:
                    messages.append({"role": "assistant", "content": "".join(buf)})

        return messages

    @staticmethod
    def _assistant_created_at(items: Iterable[object]) -> datetime:
        times = [to_aware_utc(it.created_at) for it in items]
        last_ts = max(times) if times else datetime.min.replace(tzinfo=timezone.utc)
        return max(datetime.now(timezone.utc), last_ts + timedelta(microseconds=1))

    @staticmethod
    def _assistant_start_events(
        thread: ThreadMetadata,
        item_id: str,
        created_at: datetime,
    ) -> list[ThreadStreamEvent]:
        item = AssistantMessageItem(
            id=item_id,
            thread_id=thread.id,
            created_at=created_at,
            content=[],
        )
        return [
            ThreadItemAddedEvent(item=item),
            ThreadItemUpdatedEvent(
                item_id=item_id,
                update=AssistantMessageContentPartAdded(
                    content_index=0,
                    content=AssistantMessageContent(text=""),
                ),
            ),
        ]

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

        seen_ids = {it.id for it in thread_items_page.data}
        messages_for_graph = self._extract_text_messages(thread_items_page.data)

        if input_user_message is not None and input_user_message.id not in seen_ids:
            messages_for_graph.extend(self._extract_text_messages([input_user_message]))

        assistant_created_at = self._assistant_created_at(thread_items_page.data)
        assistant_item_id = self.store.generate_item_id("message", thread, context)

        config = create_config(
            thread_id=thread.id,
            langfuse_handler=self.langfuse_handler,
        )
        last = messages_for_graph[-1]["content"] if messages_for_graph else ""

        assistant_started = False
        full_text: list[str] = []

        async for msg_type, delta in stream_graph_updates(
            last, self.graph, config, custom=True
        ):
            if not delta:
                continue

            if msg_type == "custom" and not assistant_started:
                yield ProgressUpdateEvent(icon="search", text=delta)
                continue

            if msg_type != "messages":
                continue

            if not assistant_started:
                assistant_started = True
                for ev in self._assistant_start_events(
                    thread=thread,
                    item_id=assistant_item_id,
                    created_at=assistant_created_at,
                ):
                    yield ev

            full_text.append(delta)
            yield ThreadItemUpdatedEvent(
                item_id=assistant_item_id,
                update=AssistantMessageContentPartTextDelta(content_index=0, delta=delta),
            )

        if not assistant_started:
            for ev in self._assistant_start_events(
                thread=thread,
                item_id=assistant_item_id,
                created_at=assistant_created_at,
            ):
                yield ev

        content = AssistantMessageContent(text="".join(full_text))

        yield ThreadItemUpdatedEvent(
            item_id=assistant_item_id,
            update=AssistantMessageContentPartDone(
                content_index=0,
                content=content,
            ),
        )

        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id=assistant_item_id,
                thread_id=thread.id,
                created_at=assistant_created_at,
                content=[content],
            )
        )
