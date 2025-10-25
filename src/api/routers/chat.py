from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graphs.chat import create_graph, stream_graph_updates, create_db_agent


class ClientMessage(BaseModel):
    thread_id: str
    content: str

router = APIRouter(
    prefix="/graphs",
    tags=["graphs"],
    responses={404: {"description": "Not found"}},
)


graph = create_graph()


agent_graph = create_db_agent()

@router.post("/chatbot")
async def chatbot(message: ClientMessage):
    return StreamingResponse(
        stream_graph_updates(message.content, graph, message.thread_id),
        media_type="text/event-stream",
    )


@router.post("/agent")
async def agent(message: ClientMessage):
    return StreamingResponse(
        stream_graph_updates(message.content, agent_graph, message.thread_id),
        media_type="text/event-stream",
    )
