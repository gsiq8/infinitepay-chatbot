"""
Minimal FastAPI app for Railway deployment
This version can start even if some dependencies are missing
"""

from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="InfinitePay AI Chatbot API (Minimal)",
    description="Minimal version for Railway deployment",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    conversation_id: str

@app.get("/")
def read_root():
    return {
        "message": "InfinitePay AI Chatbot API (Minimal) is running!",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "pong"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "app": "running",
            "database": "disabled",
            "embeddings": "disabled"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Minimal chat endpoint"""
    return ChatResponse(
        response="Olá! Sou o assistente da InfinitePay. Estou funcionando em modo minimal. Como posso ajudar?",
        sources=[],
        conversation_id=request.conversation_id or "new"
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting minimal server on 0.0.0.0:{port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    ) 