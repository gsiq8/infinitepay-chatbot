"""
FastAPI RAG Backend for InfinitePay AI Chatbot
Integrates Supabase vector search with Hugging Face for embeddings and text generation.
Version: 1.2.0 (HF Inference only)
"""

# CRITICAL: Load environment variables FIRST, before any other imports
from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import time
import httpx
import asyncio
from supabase import create_client, Client
from huggingface_hub import InferenceClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="InfinitePay AI Chatbot API",
    description="RAG-powered chatbot for InfinitePay customer support, using Supabase and Hugging Face.",
    version="1.2.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://giovanasiquieroli.com.br",
        "https://giovanasiquieroli.com",
        "https://www.giovanasiquieroli.com.br",
        "https://www.giovanasiquieroli.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    conversation_id: str

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]

# --- Global Variables ---
supabase_client: Client = None
hf_client: InferenceClient = None

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global supabase_client, hf_client

    logger.info("🚀 Starting InfinitePay AI Chatbot v1.2.0")

    # Environment variable check
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    hf_token = os.getenv('HF_TOKEN')

    logger.info("🔍 Environment check:")
    logger.info(f"   SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Not set'}")
    logger.info(f"   SUPABASE_KEY: {'✅ Set' if supabase_key else '❌ Not set'}")
    logger.info(f"   HF_TOKEN: {'✅ Set' if hf_token else '❌ Not set'}")

    # Initialize Supabase
    if supabase_url and supabase_key:
        try:
            supabase_client = create_client(supabase_url, supabase_key)
            logger.info("✅ Supabase client initialized.")
        except Exception as e:
            logger.error(f"❌ Supabase initialization failed: {e}")
            supabase_client = None
    else:
        logger.warning("⚠️ Supabase credentials not found. Database features will be disabled.")

    # Initialize Hugging Face Inference Client
    if hf_token:
        try:
            hf_client = InferenceClient(
                provider="hf-inference",
                api_key=hf_token
            )
            logger.info("✅ Hugging Face Inference Client initialized.")
        except Exception as e:
            logger.error(f"❌ Hugging Face Inference Client initialization failed: {e}")
            hf_client = None
    else:
        logger.warning("⚠️ HF_TOKEN not found. AI features will be disabled.")

    logger.info("🎉 Services initialization complete.")

# --- Security Filter ---
class SecurityFilter:
    """Filter out potentially harmful queries (prompt injection)."""
    BLOCKED_PATTERNS = [
        "ignore previous instructions", "ignore all instructions", "you are now",
        "pretend to be", "act as if", "forget everything", "sistema prompt", "prompt injection"
    ]

    @staticmethod
    def is_safe_query(query: str) -> bool:
        query_lower = query.lower()
        return not any(pattern in query_lower for pattern in SecurityFilter.BLOCKED_PATTERNS)

# --- RAG Service ---
class RAGService:
    """Handles Retrieval-Augmented Generation logic."""

    @staticmethod
    async def get_embedding(text: str) -> Optional[List[float]]:
        """Generates an embedding for text using Hugging Face Inference API."""
        if not hf_client:
            logger.warning("Hugging Face client not available for embedding.")
            return None
        try:
            # Use the sentence-transformers model for embeddings
            embedding = await asyncio.to_thread(
                hf_client.feature_extraction,
                text,
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            return embedding
        except Exception as e:
            logger.error(f"❌ Error getting embedding from Hugging Face: {e}")
            return None

    @staticmethod
    async def similarity_search(query: str, k: int = 5) -> List[Dict]:
        """Performs vector similarity search in Supabase."""
        if not supabase_client:
            logger.warning("Supabase client not available for similarity search.")
            return []

        query_embedding = await RAGService.get_embedding(query)
        if not query_embedding:
            logger.error("Could not generate query embedding. Aborting search.")
            return []

        try:
            result = supabase_client.rpc(
                'similarity_search',
                {'query_embedding': query_embedding, 'match_count': k}
            ).execute()
            logger.info(f"✅ Similarity search returned {len(result.data or [])} results.")
            return result.data or []
        except Exception as e:
            logger.error(f"❌ Error in similarity search RPC call: {e}")
            return []

    @staticmethod
    async def generate_response(query: str, context_docs: List[Dict]) -> str:
        """Generates a response using Hugging Face Inference API."""
        try:
            context = "\n\n".join([
                f"Documento: {doc.get('page_title', 'Sem título')}\n{doc['content']}"
                for doc in context_docs[:3]
            ])
            
            prompt = f"""Você é um assistente da InfinitePay. Responda à pergunta do cliente baseando-se APENAS no contexto abaixo. Se a informação não estiver no contexto, diga que não sabe.

Contexto:
{context}

Pergunta: {query}

Resposta:"""

            if not hf_client:
                logger.error("❌ Hugging Face client not available")
                return "Desculpe, o serviço de IA está temporariamente indisponível."

            try:
                # Use the SmolLM3-3B model for text generation
                response = await asyncio.to_thread(
                    hf_client.text_generation,
                    prompt,
                    model="HuggingFaceTB/SmolLM3-3B",
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True
                )
                logger.info("✅ Generated response using Hugging Face SmolLM3-3B")
                return response
            except Exception as hf_error:
                logger.error(f"❌ Hugging Face generation failed: {hf_error}")
                # Fallback to simple response
                if context_docs:
                    return f"Com base nas informações disponíveis, posso ajudar com sua pergunta sobre a InfinitePay. {query}"
                else:
                    return "Olá! Sou o assistente da InfinitePay. Como posso ajudar você hoje?"
                    
        except Exception as e:
            logger.error(f"❌ Error generating LLM response: {e}")
            return "Desculpe, ocorreu um erro interno ao processar sua pergunta."

# --- Health Check and Utility Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint with basic API info."""
    return {"message": "InfinitePay AI Chatbot API is running!", "status": "healthy", "version": "1.2.0"}

@app.get("/ping")
def ping():
    """Simple ping for liveness probes."""
    return {"status": "ok", "message": "pong"}

@app.get("/railway-health")
def railway_health():
    """Railway-specific health check."""
    return {"status": "healthy", "timestamp": time.time(), "app": "InfinitePay AI Chatbot"}

@app.get("/test")
def test():
    """Endpoint for debugging deployment environment variables."""
    return {
        "status": "ok", "timestamp": time.time(),
        "environment": {
            "port": os.environ.get("PORT", "8000"),
            "railway_environment": os.environ.get("RAILWAY_ENVIRONMENT", "not_set")
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check for all dependent services."""
    services = {"app": "running", "database": "unknown", "embeddings": "unknown", "llm": "unknown"}
    
    # Check services concurrently
    async def check_db():
        if supabase_client:
            try:
                await asyncio.to_thread(supabase_client.table('documents').select('id', count='exact').limit(0).execute)
                services["database"] = "healthy"
            except Exception: services["database"] = "error"
        else: services["database"] = "disabled"

    async def check_embeddings():
        if hf_client:
            try:
                await RAGService.get_embedding("health check")
                services["embeddings"] = "healthy"
            except Exception: services["embeddings"] = "error"
        else: services["embeddings"] = "disabled"

    async def check_llm():
        # The LLM service is now Hugging Face Inference, so we check its availability
        if hf_client:
            services["llm"] = "healthy"
        else:
            services["llm"] = "disabled"

    await asyncio.gather(check_db(), check_embeddings(), check_llm())
    
    overall_status = "healthy" if all(s in ["healthy", "disabled"] for s in services.values()) else "degraded"
    return HealthResponse(status=overall_status, services=services)

# --- Main API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint to get a RAG-powered response."""
    if not SecurityFilter.is_safe_query(request.query):
        raise HTTPException(status_code=400, detail="Query appears to be a prompt injection attempt.")

    relevant_docs = await RAGService.similarity_search(request.query)
    if not relevant_docs:
        logger.warning(f"No relevant documents found for query: '{request.query}'")
        return ChatResponse(
            response="Desculpe, não encontrei informações relevantes para sua pergunta. Você pode tentar reformular.",
            sources=[],
            conversation_id=request.conversation_id or "new"
        )

    response_text = await RAGService.generate_response(request.query, relevant_docs)
    sources = [
        {"title": doc.get('page_title', 'Documento'), "url": doc.get('page_url', ''), "similarity": doc.get('similarity', 0)}
        for doc in relevant_docs[:3]
    ]

    return ChatResponse(
        response=response_text,
        sources=sources,
        conversation_id=request.conversation_id or "new"
    )

@app.get("/search")
async def search_documents(query: str, limit: int = 10):
    """Endpoint for testing document retrieval."""
    if not SecurityFilter.is_safe_query(query):
        raise HTTPException(status_code=400, detail="Query appears to be a prompt injection attempt.")
    
    results = await RAGService.similarity_search(query, limit)
    return {"results": results}

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")