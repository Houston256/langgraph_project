from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from assistant.api.routers import chat, eval, ui

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
