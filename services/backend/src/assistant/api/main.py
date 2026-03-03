import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from assistant.api.routers import chat, eval, ui

app = FastAPI()

cors_origins = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(chat.router)
app.include_router(ui.router)
app.include_router(eval.router)


@app.get("/")
async def root():
    return {"message": "Hello From Root!"}
