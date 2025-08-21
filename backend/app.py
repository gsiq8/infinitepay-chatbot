"""
Ultra-minimal FastAPI app for Railway deployment
No complex dependencies, no startup events, just basic functionality
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import time

# Initialize FastAPI app
app = FastAPI(
    title="InfinitePay AI Chatbot API",
    description="RAG-powered chatbot for InfinitePay customer support",
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

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "InfinitePay AI Chatbot API is running!",
        "status": "healthy",
        "version": "1.0.0"
    }

# Health check endpoint
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

# Simple ping endpoint
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "pong"}

# Railway health check endpoint
@app.get("/railway-health")
def railway_health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "app": "InfinitePay AI Chatbot",
        "version": "1.0.0"
    }

# Test endpoint for debugging
@app.get("/test")
def test():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "environment": {
            "port": os.environ.get("PORT", "8000"),
            "railway_environment": os.environ.get("RAILWAY_ENVIRONMENT", "not_set")
        }
    }

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    return ChatResponse(
        response="Olá! Sou o assistente da InfinitePay. Como posso ajudar você hoje?",
        sources=[],
        conversation_id=request.conversation_id or "new"
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on 0.0.0.0:{port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    ) 