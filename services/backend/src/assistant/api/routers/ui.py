import os

from chatkit.server import StreamingResult
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from assistant.ui.server import LangGraphChatKitServer
from assistant.ui.store import MemoryStore

router = APIRouter(
    prefix="/ui",
    tags=["ui"],
    responses={404: {"description": "Not found"}},
)

data_store = MemoryStore()
delta_coalesce_ms = float(os.environ.get("DELTA_COALESCE_MS", "100")) # without coalescing, ChatKit can't keep up with
server = LangGraphChatKitServer(data_store, delta_coalesce_interval_ms=delta_coalesce_ms)


@router.post(
    "/chat",
)
async def chatkit_endpoint(request: Request):
    result = await server.process(await request.body(), {})
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    return Response(content=result.json, media_type="application/json")
