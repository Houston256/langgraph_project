from fastapi import FastAPI

from .routers import chat

app = FastAPI()

app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "Hello From Root!"}