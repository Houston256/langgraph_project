import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from assistant.api.identity import TrustedHeaderIdentity
from assistant.api.routers import runs, threads
from assistant.config import settings
from assistant.db import Listener, open_db
from assistant.graphs.registry import open_graphs
from assistant.runs.events import PgEventLog


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build everything the handlers need once, and hand it over on app.state."""
    listener = Listener(["run_events"])
    async with open_db() as (pool, checkpointer), open_graphs(checkpointer) as graphs:
        app.state.pool = pool
        app.state.checkpointer = checkpointer
        app.state.graphs = graphs
        app.state.events = PgEventLog(pool, listener)
        app.state.identity = TrustedHeaderIdentity(
            settings.identity_header, settings.default_user_id
        )
        listen = asyncio.create_task(listener.run())
        try:
            yield
        finally:
            listen.cancel()


app = FastAPI(lifespan=lifespan)

app.include_router(threads.router, prefix="/api/v1")
app.include_router(runs.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "OK"}
