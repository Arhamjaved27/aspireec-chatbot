from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
from app.session_manager import get_session, add_message
from app.chatbot_core import initialize_resources, answer_query
import os


app = FastAPI()

# CORS for iframe embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


frontend_path = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory="C:/Users/Arham/Desktop/aspireec-chatbot/static"), name="static")

# Serve frontend
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

# Request/response models
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    messages: list

# Load resources
chunks, vectorizer = initialize_resources()

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = get_session(req.session_id)
    messages.append({"role": "user", "content": req.message})
    context, reply = answer_query(req.message, messages, chunks, vectorizer)
    messages.append({"role": "assistant", "content": reply})
    add_message(req.session_id, messages)
    return ChatResponse(response=reply, messages=messages)
