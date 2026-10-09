from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from assistant.api.identity import current_user
from assistant.config import settings
from assistant.graphs.registry import DEFAULT_GRAPH
from assistant.runs import store
from assistant.runs.messages import to_ui_messages
from assistant.usage import limit_exceeded

router = APIRouter(prefix="/threads", tags=["threads"])


class Thread(BaseModel):
    thread_id: UUID
    graph: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    active_run_id: UUID | None


class ThreadWithMessages(Thread):
    messages: list[dict[str, Any]]


class CreateThread(BaseModel):
    graph: str = DEFAULT_GRAPH


class RenameThread(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class CreateRun(BaseModel):
    content: str = Field(min_length=1)


class Run(BaseModel):
    run_id: UUID


async def owned_thread(thread_id: UUID, request: Request, user: str) -> dict[str, Any]:
    thread = await store.get_thread(request.app.state.pool, thread_id, user)
    if thread is None:
        raise HTTPException(404, "thread not found")
    return thread


@router.post("")
async def create_thread(
    body: CreateThread, request: Request, user: str = Depends(current_user)
) -> Thread:
    if body.graph not in request.app.state.graphs:
        raise HTTPException(422, f"unknown graph {body.graph!r}")
    return Thread(**await store.create_thread(request.app.state.pool, user, body.graph))


@router.get("")
async def list_threads(
    request: Request, user: str = Depends(current_user)
) -> list[Thread]:
    return [Thread(**t) for t in await store.list_threads(request.app.state.pool, user)]


@router.get("/{thread_id}")
async def get_thread(
    thread_id: UUID, request: Request, user: str = Depends(current_user)
) -> ThreadWithMessages:
    thread = await owned_thread(thread_id, request, user)
    if (spec := request.app.state.graphs.get(thread["graph"])) is None:
        raise HTTPException(404, f"graph {thread['graph']!r} no longer exists")
    graph = spec.graph
    state = await graph.aget_state({"configurable": {"thread_id": str(thread_id)}})
    return ThreadWithMessages(
        **thread, messages=to_ui_messages(state.values.get("messages", []))
    )


@router.patch("/{thread_id}")
async def rename_thread(
    thread_id: UUID,
    body: RenameThread,
    request: Request,
    user: str = Depends(current_user),
) -> Thread:
    if not await store.rename_thread(
        request.app.state.pool, thread_id, user, body.title
    ):
        raise HTTPException(404, "thread not found")
    return Thread(**await owned_thread(thread_id, request, user))


@router.delete("/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: UUID, request: Request, user: str = Depends(current_user)
) -> None:
    pool = request.app.state.pool
    if not await store.delete_thread(pool, thread_id, user):
        raise HTTPException(404, "thread not found")
    # With a run still open, the worker purges the thread when the run stops.
    await store.purge_thread(pool, thread_id, request.app.state.checkpointer)


@router.post("/{thread_id}/runs", status_code=202)
async def create_run(
    thread_id: UUID,
    body: CreateRun,
    request: Request,
    user: str = Depends(current_user),
) -> Run:
    if len(body.content) > settings.max_input_chars:
        raise HTTPException(
            413, f"message longer than {settings.max_input_chars} characters"
        )
    await owned_thread(thread_id, request, user)
    if reason := await limit_exceeded(request.app.state.pool, user):
        raise HTTPException(429, reason)
    run_id = await store.create_run(request.app.state.pool, thread_id, body.content)
    if run_id is None:
        raise HTTPException(404, "thread not found")
    return Run(run_id=run_id)
