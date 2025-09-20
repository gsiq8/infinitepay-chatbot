"""
FastAPI RAG Backend for InfinitePay AI Chatbot
Simple and direct RAG implementation following Supabase best practices.
Version: 2.0.0 (Simplified RAG)
"""

# CRITICAL: Load environment variables FIRST, before any other imports
from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
from supabase import create_client, Client
from huggingface_hub import InferenceClient
from huggingface_hub.utils._auth import get_token
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="InfinitePay AI Chatbot API",
    description="Simple RAG-powered chatbot using Supabase pgvector and Hugging Face.",
    version="2.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://giovanasiquieroli.com.br",
        "https://www.giovanasiquieroli.com.br"
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
client: InferenceClient = None

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global supabase_client, client

    logger.info("🚀 Starting InfinitePay AI Chatbot v2.0.0 (Simplified)")

    # Environment variable check
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    hf_token = os.getenv('HF_TOKEN') or get_token()

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
            client = InferenceClient(
                provider="auto",
                api_key=hf_token
            )
            logger.info("✅ Hugging Face Inference Client initialized.")
        except Exception as e:
            logger.error(f"❌ Hugging Face Inference Client initialization failed: {e}")
            client = None
    else:
        logger.warning("⚠️ HF_TOKEN not found. AI features will be disabled.")

    logger.info("🎉 Services initialization complete.")

# --- Simple RAG Service ---
class SimpleRAGService:
    """Simplified RAG implementation following Supabase best practices."""

    @staticmethod
    async def get_embedding(text: str) -> Optional[List[float]]:
        """Generate embedding using HuggingFace."""
        if not client:
            logger.warning("HuggingFace client not available.")
            return None
        
        try:
            logger.info(f"🔢 Generating embedding for: {text[:50]}...")
            embedding = await asyncio.to_thread(
                client.feature_extraction,
                text,
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Convert to proper format
            if isinstance(embedding, np.ndarray):
                if embedding.ndim == 2:
                    embedding = embedding.flatten()
                embedding = embedding.tolist()
            elif isinstance(embedding, list) and isinstance(embedding[0], list):
                embedding = embedding[0]
            
            # Ensure all values are floats
            embedding = [float(x) for x in embedding]
            
            logger.info(f"✅ Embedding generated: length={len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return None

    @staticmethod
    async def search_similar_documents(query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """Busca otimizada para Supabase vector"""
        if not supabase_client or not query_embedding:
            return []
        
        try:
            logger.info(f"🔍 Searching for similar documents, embedding length: {len(query_embedding)}")
            
            # Try using RPC function first (recommended approach)
            try:
                result = supabase_client.rpc(
                    'match_documents',
                    {
                        'query_embedding': query_embedding,
                        'match_count': limit,
                        'match_threshold': 0.1  # Lower threshold for more results
                    }
                ).execute()
                
                if result.data:
                    logger.info(f"✅ RPC search found {len(result.data)} documents")
                    return result.data
                else:
                    logger.warning("RPC search returned no results")
            except Exception as rpc_error:
                logger.warning(f"RPC search failed: {rpc_error}")
            
            # Fallback: Try manual search with more comprehensive query
            try:
                logger.info("Trying manual search fallback...")
                result = supabase_client.table('documents').select(
                    'id, content, page_title, page_url, metadata, chunk_index, embedding'
                ).not_.is_('embedding', 'null') \
                .eq('is_public', True) \
                .limit(500).execute()
                
                if result.data:
                    # Calculate cosine similarity for each document
                    scored_docs = []
                    for doc in result.data:
                        if doc.get('embedding'):
                            similarity = SimpleRAGService._cosine_similarity(query_embedding, doc['embedding'])
                            if similarity > 0.1:  # Filter low relevance results
                                doc['similarity'] = similarity
                                scored_docs.append(doc)
                    
                    # Sort by similarity and return top results
                    scored_docs.sort(key=lambda x: x['similarity'], reverse=True)
                    top_docs = scored_docs[:limit]
                    
                    logger.info(f"✅ Manual similarity search found {len(top_docs)} documents")
                    return top_docs
                    
            except Exception as table_error:
                logger.error(f"Direct table search failed: {table_error}")
            
            # Final fallback
            return await SimpleRAGService.text_search_fallback(limit)
                
        except Exception as e:
            logger.error(f"❌ All vector search methods failed: {e}")
            return await SimpleRAGService.text_search_fallback(query_embedding, limit)

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        a, b = np.array(vec1), np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @staticmethod
    async def text_search_fallback(limit: int) -> List[Dict]:
        """Fallback to simple text search if vector search fails."""
        try:
            logger.info("🔄 Using text search fallback...")
            # Simple search by content - for testing
            result = supabase_client.table('documents').select(
                'id, page_title, content, page_url'
            ).limit(limit).execute()
            
            if result.data:
                # Add mock similarity scores
                for doc in result.data:
                    doc['similarity'] = 0.5
                logger.info(f"✅ Fallback found {len(result.data)} documents")
                return result.data
            
        except Exception as e:
            logger.error(f"❌ Text search fallback failed: {e}")
        
        return []

    @staticmethod
    async def generate_answer(query: str, context_docs: List[Dict]) -> str:
        """Generate answer using context documents with crash protection."""
        if not client:
            logger.warning("Client not available")
            return "Desculpe, o serviço de IA está temporariamente indisponível."
    
        try:
            # Timeout protection
            import asyncio
        
            async def _generate_with_timeout():
                # Build context from documents
                context = ""
                if context_docs:
                    context_parts = []
                    logger.info(f"Processing {len(context_docs)} documents")
                
                    for i, doc in enumerate(context_docs[:3]):  # Limit to 3 docs
                        try:
                            title = str(doc.get('page_title', 'Documento'))[:100]  # Limit title length
                            content = str(doc.get('content', ''))[:600]  # Limit content length
                            similarity = float(doc.get('similarity', 0))
                        
                            logger.info(f"Doc {i+1}: {title[:50]}... (sim: {similarity:.3f})")
                            context_parts.append(f"=== {title} ===\n{content}")
                        
                        except Exception as doc_error:
                            logger.error(f"Error processing doc {i}: {doc_error}")
                            continue
                
                    context = "\n\n".join(context_parts)
                
                    # Limit total context size to prevent memory issues
                    if len(context) > 3000:
                        context = context[:3000] + "..."
                        logger.warning("Context truncated due to length")
                    
                else:
                    logger.warning("No context documents provided")
                    return "Não encontrei informações relevantes para responder sua pergunta."
            
                if not context.strip():
                    logger.warning("Empty context after processing")
                    return "Não foi possível processar as informações encontradas."
            
                # Create simple, concise system message
                system_content = f"""Responda com base apenas no contexto fornecido. Se não souber, diga que não encontrou a informação.

    Contexto:
    {context}"""
            
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query}
                ]
            
                logger.info(f"Making API call with context length: {len(context)}")
            
                try:
                    # Make the API call with timeout
                    response = client.chat.completions.create(
                        messages=messages,
                        model="Qwen/Qwen3-32B",
                        stream=False,
                        max_tokens=300,  # Shorter responses
                        temperature=0.2,
                        timeout=30  # 30 second timeout
                    )
                
                    if not response or not response.choices:
                        logger.error("Empty response from API")
                        return "Desculpe, houve um problema ao gerar a resposta."
                
                    content = response.choices[0].message.content
                    if not content:
                        logger.error("Empty content in response")
                        return "Não consegui gerar uma resposta adequada."
                
                    # Clean up response
                    content = content.strip()
                
                    # Remove reasoning tokens if present
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                
                    logger.info(f"Generated response: {content[:100]}...")
                    return content
                
                except Exception as api_error:
                    logger.error(f"API call failed: {str(api_error)}")
                    logger.error(f"API error type: {type(api_error)}")
                    return "Desculpe, o serviço está temporariamente indisponível. Tente novamente em alguns minutos."
        
            # Run with timeout to prevent hanging
            try:
                result = await asyncio.wait_for(_generate_with_timeout(), timeout=45.0)
                return result
            except asyncio.TimeoutError:
                logger.error("Request timed out")
                return "Desculpe, a consulta demorou muito para ser processada. Tente novamente."
            
        except Exception as outer_error:
            # Catch-all to prevent crashes
            logger.error(f"Critical error in generate_answer: {str(outer_error)}")
            logger.error(f"Error type: {type(outer_error)}")
        
            # Don't raise the exception - return a safe fallback
            try:
                return await SimpleRAGService._create_fallback_response(query, context_docs)
            except Exception as fallback_error:
                logger.error(f"Even fallback failed: {fallback_error}")
                return "Desculpe, estou com dificuldades técnicas no momento. Tente novamente mais tarde."
    
    @staticmethod
    async def _create_fallback_response(query: str, context_docs: List[Dict]) -> str:
        """Create a fallback response when text generation fails."""
    
        if context_docs:
            titles = [doc.get('page_title', 'documento') for doc in context_docs[:3]]
            similarities = [f"{doc.get('similarity', 0):.1f}" for doc in context_docs[:3]]
            
            response = f"""Com base na consulta à nossa base de conhecimento sobre "{query}", encontrei informações relevantes nos seguintes documentos:\n\n"""
            for i, (title, sim) in enumerate(zip(titles, similarities), 1):
                response += f"{i}. {title} (relevância: {sim})\n"
            
            return response
        
        return "Não encontrei informações específicas sobre sua pergunta. Por favor, reformule ou entre em contato com nosso suporte."
        

# --- Health Check Endpoints ---
@app.get("/")
def read_root():
    return {"message": "InfinitePay AI Chatbot API v2.0.0", "status": "healthy"}

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "pong"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check for all services."""
    services = {"app": "running"}
    
    # Check Supabase
    if supabase_client:
        try:
            await asyncio.to_thread(
                supabase_client.table('documents').select('id', count='exact').limit(1).execute
            )
            services["database"] = "healthy"
        except Exception:
            services["database"] = "error"
    else:
        services["database"] = "disabled"
    
    # Check HuggingFace
    if client:
        services["ai"] = "healthy"
    else:
        services["ai"] = "disabled"
    
    status = "healthy" if all(s in ["healthy", "disabled"] for s in services.values()) else "degraded"
    return HealthResponse(status=status, services=services)

# --- Main Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - simplified RAG."""
    logger.info(f"💬 Chat request: {request.query}")
    
    # 1. Get embedding for the query
    query_embedding = await SimpleRAGService.get_embedding(request.query)
    if not query_embedding:
        return ChatResponse(
            response="Desculpe, não consegui processar sua pergunta no momento.",
            sources=[],
            conversation_id=request.conversation_id or "new"
        )
    
    # 2. Search for similar documents
    similar_docs = await SimpleRAGService.search_similar_documents(query_embedding, limit=5)
    
    # 3. Generate answer
    answer = await SimpleRAGService.generate_answer(request.query, similar_docs)
    
    # 4. Prepare sources
    sources = []
    for doc in similar_docs[:3]:
        sources.append({
            "title": doc.get('page_title', 'Documento'),
            "url": doc.get('page_url', ''),
            "similarity": round(doc.get('similarity', 0), 3)
        })
    
    return ChatResponse(
        response=answer,
        sources=sources,
        conversation_id=request.conversation_id or "new"
    )

# --- Debug Endpoints ---
@app.get("/debug/embedding")
async def debug_embedding(text: str = "Como funcionam os planos?"):
    """Test embedding generation."""
    embedding = await SimpleRAGService.get_embedding(text)
    return {
        "text": text,
        "embedding_length": len(embedding) if embedding else 0,
        "embedding_sample": embedding[:5] if embedding else None,
        "success": embedding is not None
    }

@app.get("/debug/documents")
async def debug_documents(limit: int = 5):
    """Check if documents exist in database and have embeddings."""
    if not supabase_client:
        return {"error": "Supabase not available"}
    
    try:
        # Check total documents
        count_result = supabase_client.table('documents').select('id', count='exact').execute()
        total_docs = count_result.count if hasattr(count_result, 'count') else 0
        
        # Get sample documents with embedding info
        result = supabase_client.table('documents').select(
            'id, page_title, content, embedding'
        ).limit(limit).execute()
        
        documents = []
        docs_with_embeddings = 0
        
        for doc in result.data or []:
            has_embedding = doc.get('embedding') is not None
            if has_embedding:
                docs_with_embeddings += 1
            
            documents.append({
                "id": doc.get('id'),
                "title": doc.get('page_title', 'No title'),
                "content_preview": doc.get('content', '')[:100] + "..." if doc.get('content') else "No content",
                "has_embedding": has_embedding,
                "embedding_length": len(doc.get('embedding', [])) if has_embedding else 0
            })
        
        return {
            "total_documents": total_docs,
            "sample_size": len(documents),
            "documents_with_embeddings": docs_with_embeddings,
            "documents": documents,
            "needs_embeddings": docs_with_embeddings == 0
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/generate-embeddings")
async def generate_embeddings_for_documents(limit: int = 10, force: bool = False):
    """Generate embeddings for documents that don't have them yet."""
    if not supabase_client or not client:
        raise HTTPException(status_code=503, detail="Required services not available")
    
    try:
        # Get documents without embeddings (or all if force=True)
        if force:
            query_builder = supabase_client.table('documents').select('id, content, page_title')
        else:
            query_builder = supabase_client.table('documents').select('id, content, page_title').is_('embedding', 'null')
        
        result = query_builder.limit(limit).execute()
        documents = result.data or []
        
        if not documents:
            return {"message": "No documents need embeddings", "processed": 0}
        
        processed = 0
        errors = 0
        
        for doc in documents:
            try:
                # Create text to embed (title + content)
                text_to_embed = f"{doc.get('page_title', '')} {doc.get('content', '')}"
                text_to_embed = text_to_embed.strip()[:1000]  # Limit length
                
                if not text_to_embed:
                    continue
                
                # Generate embedding
                embedding = await SimpleRAGService.get_embedding(text_to_embed)
                
                if embedding:
                    # Update document with embedding
                    supabase_client.table('documents').update({
                        'embedding': embedding
                    }).eq('id', doc['id']).execute()
                    
                    processed += 1
                    logger.info(f"✅ Generated embedding for document {doc['id']}")
                else:
                    errors += 1
                    logger.error(f"❌ Failed to generate embedding for document {doc['id']}")
                
            except Exception as doc_error:
                errors += 1
                logger.error(f"❌ Error processing document {doc.get('id')}: {doc_error}")
                continue
        
        return {
            "message": f"Processed {processed} documents",
            "processed": processed,
            "errors": errors,
            "total_documents": len(documents)
        }
        
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")