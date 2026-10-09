import json
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from assistant.api.identity import current_user
from assistant.runs import store
from assistant.runs.events import EventLog
from assistant.runs.store import RunStatus

router = APIRouter(prefix="/runs", tags=["runs"])

# Keeps proxies from closing an idle stream.
PING_SECONDS = 15.0


@router.post("/{run_id}/cancel", status_code=202)
async def cancel_run(
    run_id: UUID, request: Request, user: str = Depends(current_user)
) -> None:
    if not await store.cancel_run(request.app.state.pool, run_id, user):
        raise HTTPException(409, "run not found or not active")


def sse(seq: int | None, data: dict) -> str:
    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}data: {json.dumps(data)}\n\n"


@router.get("/{run_id}/stream")
async def stream_run(
    run_id: UUID,
    request: Request,
    user: str = Depends(current_user),
    last_event_id: int | None = Header(None),
    after: int | None = None,
) -> StreamingResponse:
    """Run events as SSE, from seq > after (or Last-Event-ID) until run-end."""
    pool = request.app.state.pool
    events: EventLog = request.app.state.events
    run = await store.get_run(pool, run_id, user)
    if run is None:
        raise HTTPException(404, "run not found")
    cursor = after if after is not None else (last_event_id or 0)

    async def ended() -> RunStatus | None:
        """The run's status if it ended and no events remain to send."""
        status = await store.run_status(pool, run_id)
        if status is None:
            return RunStatus.ERROR
        if status in store.TERMINAL and await events.last_seq(run_id) <= cursor:
            return status
        return None

    async def body():
        nonlocal cursor
        if status := await ended():
            # Events already trimmed; the client reloads the thread.
            yield sse(None, {"type": "run-end", "status": status})
            return
        async for item in events.read(run_id, cursor, PING_SECONDS):
            if item is not None:
                cursor = item[0]
                yield sse(*item)
                continue
            if status := await ended():
                yield sse(None, {"type": "run-end", "status": status})
                return
            yield ": ping\n\n"

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
