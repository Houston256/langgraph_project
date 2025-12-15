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
server = LangGraphChatKitServer(data_store)


@router.post(
    "/chat",
)
async def chatkit_endpoint(request: Request):
    result = await server.process(await request.body(), {})
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    return Response(content=result.json, media_type="application/json")
