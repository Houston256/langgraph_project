"""Runs one claimed run's graph and turns its output into events."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from assistant.config import settings
from assistant.db import Pool
from assistant.graphs.registry import GraphSpec
from assistant.runs.events import Emitter, EventLog
from assistant.runs.links import LinkFilter, sanitize
from assistant.runs.messages import cancel_updates, human_message_id, ui_artifact
from assistant.usage import Pricing, UsageRecorder, limits_enabled
from assistant.utils.streaming import create_config, normalize_delta

TitleMaker = Callable[
    [Pool, dict[str, Any], list[BaseCallbackHandler]], Awaitable[None]
]


class RunExecutor:
    def __init__(
        self,
        pool: Pool,
        events: EventLog,
        graphs: dict[str, GraphSpec],
        pricing: Pricing,
        make_title: TitleMaker,
    ):
        self.pool = pool
        self.events = events
        self.graphs = graphs
        self.pricing = pricing
        self.make_title = make_title
        # Keeps title tasks referenced until they finish.
        self._side_tasks: set[asyncio.Task] = set()

    def config(self, run: dict[str, Any]) -> RunnableConfig:
        config = create_config(str(run["thread_id"]))
        config["callbacks"] = [
            UsageRecorder(self.pool, self.pricing, run["run_id"], run["user_id"])
        ]
        return config

    async def unpriced_models(self) -> list[str]:
        """Models without a known price, while a spending limit is on."""
        if not limits_enabled():
            return []
        await self.pricing.refresh()
        models = {settings.model_name, settings.summary_model_name}
        return sorted(m for m in models if self.pricing.get(m) is None)

    async def stream(self, run: dict[str, Any], emitter: Emitter) -> None:
        spec = self.graphs[run["graph"]]
        config = self.config(run)
        hid = human_message_id(run["run_id"])

        state = await spec.graph.aget_state(config)
        messages = state.values.get("messages", [])
        started = any(m.id == hid for m in messages)
        if run["attempt"] > 1:
            await self.resend_checkpointed(emitter, messages, hid)
        if started and not state.next:
            # Finished before the last worker could mark it done.
            return
        if not started and run["title"] is None:
            task = asyncio.create_task(
                self.make_title(self.pool, run, config["callbacks"])
            )
            self._side_tasks.add(task)
            task.add_done_callback(self._side_tasks.discard)
        # None resumes from the last checkpoint after a crash.
        graph_input = (
            None
            if started
            else {"messages": [HumanMessage(content=run["input"]["content"], id=hid)]}
        )

        # Off-site links are cut out before text reaches the event log.
        links = LinkFilter()
        text_id: str | None = None

        async def flush_text() -> None:
            if text_id and (rest := links.flush()):
                await emitter.emit(
                    {"type": "text-delta", "message_id": text_id, "delta": rest}
                )

        async for mode, payload in spec.graph.astream(
            graph_input,  # type: ignore[arg-type]
            config=config,
            stream_mode=["messages", "updates", "custom"],
        ):
            if mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") != spec.text_node or not isinstance(
                    chunk, AIMessageChunk
                ):
                    continue
                if chunk.id != text_id:
                    await flush_text()
                    text_id = chunk.id
                if delta := links.feed(normalize_delta(chunk.content)):
                    await emitter.emit(
                        {"type": "text-delta", "message_id": text_id, "delta": delta}
                    )
            elif mode == "custom":
                if isinstance(payload, str):
                    await flush_text()
                    await emitter.emit({"type": "status", "text": payload})
            elif mode == "updates":
                await flush_text()
                for update in (payload or {}).values():
                    updated = (
                        update.get("messages") if isinstance(update, dict) else None
                    )
                    for m in updated if isinstance(updated, list) else [updated]:
                        if ui := ui_artifact(m):
                            await emitter.emit({"type": "ui", **ui})
        await flush_text()

    async def resend_checkpointed(
        self, emitter: Emitter, messages: list[BaseMessage], hid: str
    ) -> None:
        """After a retry or crash: tell clients to drop the live answer, then resend what the
        checkpoint kept of this run, so text of the interrupted call isn't shown twice."""
        await emitter.emit({"type": "reset"})
        ids = [m.id for m in messages]
        for m in messages[ids.index(hid) + 1 :] if hid in ids else []:
            if isinstance(m, AIMessage) and (
                text := sanitize(normalize_delta(m.content))
            ):
                await emitter.emit(
                    {"type": "text-delta", "message_id": m.id, "delta": text}
                )
            elif isinstance(m, ToolMessage) and (ui := ui_artifact(m)):
                await emitter.emit({"type": "ui", **ui})
        await emitter.flush()

    async def finalize_cancel(self, run: dict[str, Any]) -> None:
        """Makes the thread match what the user saw of the cancelled run."""
        spec = self.graphs[run["graph"]]
        config = self.config(run)
        state = await spec.graph.aget_state(config)
        updates = cancel_updates(
            state.values.get("messages", []),
            run["run_id"],
            run["input"]["content"],
            await self.events.all(run["run_id"]),
        )
        if updates:
            await spec.graph.aupdate_state(
                config, {"messages": updates}, as_node=spec.text_node
            )
