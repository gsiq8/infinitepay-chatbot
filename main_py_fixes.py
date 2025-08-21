# Fixes for main.py RAG issues

import logging
import os
import asyncio
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from huggingface_hub import InferenceClient
from supabase import create_client, Client
from pydantic import BaseModel

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global clients
supabase_client: Optional[Client] = None
hf_client: Optional[InferenceClient] = None

# Define FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    global supabase_client, hf_client
    logger.info("🚀 Starting InfinitePay AI Chatbot v1.2.0")
    try:
        await initialize_services()
        yield
    finally:
        logger.info("🔄 Shutting down services...")
        if supabase_client:
            supabase_client = None
        if hf_client:
            hf_client = None

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

class RAGService:
    """RAG (Retrieval-Augmented Generation) service for handling embeddings and responses."""
    
    @staticmethod
    async def generate_response(query: str, relevant_docs: List[Dict]) -> str:
        """Generate a response using the query and relevant documents."""
        try:
            # Prepare context from relevant documents
            context = "\n".join([doc.get('content', '') for doc in relevant_docs])
            
            # Add your response generation logic here
            # This could be using an LLM or other generation method
            return f"Aqui está uma resposta baseada em {len(relevant_docs)} documentos relevantes."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Desculpe, ocorreu um erro ao gerar a resposta."

# Define models
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]
    conversation_id: str

class SecurityFilter:
    @staticmethod
    def is_safe_query(query: str) -> bool:
        # Add your security filtering logic here
        return True

# 1. Fix the get_embedding method in RAGService class
@staticmethod
async def get_embedding(text: str) -> Optional[List[float]]:
    """Generates an embedding for text using Hugging Face Inference API."""
    if not hf_client:
        logger.warning("Hugging Face client not available for embedding.")
        return None
    try:
        logger.info(f"Getting embedding for text: {text[:50]}...")
        # Use the sentence-transformers model for embeddings
        embedding = await asyncio.to_thread(
            hf_client.feature_extraction,
            text,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info(f"✅ Embedding generated successfully, length: {len(embedding) if embedding else 0}")
        
        # FIX 1: Convert numpy array to list if needed
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        return embedding
    except Exception as e:
        logger.error(f"❌ Error getting embedding from Hugging Face: {e}")
        return None

# 2. Fix the similarity_search method in RAGService class  
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
        logger.info(f"Performing similarity search for query: '{query}' with embedding length: {len(query_embedding)}")
        
        # FIX 2: Ensure embedding is a list (already handled in get_embedding now)
        embedding_list = query_embedding if isinstance(query_embedding, list) else query_embedding.tolist()
        
        result = supabase_client.rpc(
            'similarity_search',
            {'query_embedding': embedding_list, 'match_count': k}
        ).execute()
        
        logger.info(f"✅ Similarity search returned {len(result.data or [])} results.")
        
        # FIX 3: Add more detailed logging
        if result.data:
            similarities = [doc.get('similarity', 0) for doc in result.data]
            logger.info(f"📊 Similarity scores: min={min(similarities):.4f}, max={max(similarities):.4f}")
            
            # FIX 4: Check if similarities are too low
            best_similarity = max(similarities)
            if best_similarity < 0.2:  # Adjust threshold as needed
                logger.warning(f"⚠️ Low similarity scores. Best: {best_similarity:.4f}")
        
        logger.info(f"Raw result: {result}")
        return result.data or []
    except Exception as e:
        logger.error(f"❌ Error in similarity search RPC call: {e}")
        # FIX 5: Try to get some basic info about the documents table
        try:
            logger.info("Trying to get basic document count...")
            basic_result = supabase_client.table('documents').select('id', count='exact').limit(1).execute()
            logger.info(f"Basic query result: {basic_result}")
            
            # Check if we have embeddings in the database
            embedding_check = supabase_client.table('documents').select('id, embedding').limit(1).execute()
            if embedding_check.data and embedding_check.data[0].get('embedding'):
                db_embedding_dim = len(embedding_check.data[0]['embedding'])
                query_embedding_dim = len(query_embedding)
                logger.info(f"📐 DB embedding dim: {db_embedding_dim}, Query embedding dim: {query_embedding_dim}")
                
                if db_embedding_dim != query_embedding_dim:
                    logger.error(f"❌ EMBEDDING DIMENSION MISMATCH! DB: {db_embedding_dim}, Query: {query_embedding_dim}")
            else:
                logger.error("❌ No embeddings found in database!")
                
        except Exception as basic_error:
            logger.error(f"❌ Even basic query failed: {basic_error}")
        return []

# 3. Enhanced chat endpoint with better error handling
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint to get a RAG-powered response."""
    if not SecurityFilter.is_safe_query(request.query):
        raise HTTPException(status_code=400, detail="Query appears to be a prompt injection attempt.")

    logger.info(f"🔍 Processing query: '{request.query}'")
    
    # FIX 6: Add more detailed logging for the chat process
    try:
        relevant_docs = await RAGService.similarity_search(request.query, k=10)  # Increased k for better results
        
        if not relevant_docs:
            logger.warning(f"❌ No relevant documents found for query: '{request.query}'")
            
            # FIX 7: Try with a simpler/broader query
            simple_query = " ".join(request.query.split()[:3])  # Use first 3 words
            logger.info(f"🔄 Retrying with simplified query: '{simple_query}'")
            relevant_docs = await RAGService.similarity_search(simple_query, k=5)
            
            if not relevant_docs:
                return ChatResponse(
                    response="Desculpe, não encontrei informações relevantes para sua pergunta. Você pode tentar reformular ou usar termos mais específicos como 'InfinitePay', 'pagamento', 'maquininha', etc.",
                    sources=[],
                    conversation_id=request.conversation_id or "new"
                )

        # FIX 8: Filter results by minimum similarity threshold
        min_similarity = 0.15  # Adjust this threshold
        filtered_docs = [doc for doc in relevant_docs if doc.get('similarity', 0) >= min_similarity]
        
        if not filtered_docs:
            logger.warning(f"⚠️ All results below similarity threshold {min_similarity}")
            # Use original results but mention low confidence
            filtered_docs = relevant_docs[:3]
            low_confidence = True
        else:
            low_confidence = False
            
        logger.info(f"✅ Using {len(filtered_docs)} documents for response generation")

        response_text = await RAGService.generate_response(request.query, filtered_docs)
        
        # FIX 9: Add confidence indicator to response
        if low_confidence:
            response_text = f"(Confiança baixa) {response_text}"
            
        sources = [
            {
                "title": doc.get('page_title', 'Documento'), 
                "url": doc.get('page_url', ''), 
                "similarity": round(doc.get('similarity', 0), 4)
            }
            for doc in filtered_docs[:3]
        ]

        return ChatResponse(
            response=response_text,
            sources=sources,
            conversation_id=request.conversation_id or "new"
        )
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# 4. Additional debugging endpoint
@app.get("/debug/embedding-test")
async def debug_embedding_test(query: str = "InfinitePay pagamento"):
    """Debug endpoint to test embedding generation and search"""
    try:
        # Test embedding generation
        embedding = await RAGService.get_embedding(query)
        
        if not embedding:
            return {"error": "Failed to generate embedding"}
            
        # Test similarity search
        results = await RAGService.similarity_search(query, k=5)
        
        return {
            "query": query,
            "embedding_length": len(embedding),
            "embedding_sample": embedding[:5] if len(embedding) > 5 else embedding,
            "results_count": len(results),
            "results": [
                {
                    "title": doc.get('page_title', 'No title')[:50],
                    "similarity": doc.get('similarity', 0),
                    "content_preview": doc.get('content', '')[:100]
                }
                for doc in results
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}

# 5. Database health check endpoint
@app.get("/debug/database-check")
async def debug_database_check():
    """Check database content and embeddings"""
    try:
        # Check total documents
        total_docs = supabase_client.table('documents').select('id', count='exact').limit(0).execute()
        
        # Check documents with embeddings
        docs_with_embeddings = supabase_client.table('documents').select('id', count='exact').not_.is_('embedding', 'null').execute()
        
        # Check sample content
        sample_docs = supabase_client.table('documents').select('id, page_title, content, embedding').limit(5).execute()
        
        sample_info = []
        for doc in sample_docs.data:
            sample_info.append({
                "id": doc.get('id'),
                "title": doc.get('page_title', 'No title')[:50],
                "content_length": len(doc.get('content', '')),
                "has_embedding": doc.get('embedding') is not None,
                "embedding_dim": len(doc.get('embedding', [])) if doc.get('embedding') else 0
            })
            
        return {
            "total_documents": total_docs.count,
            "documents_with_embeddings": docs_with_embeddings.count,
            "sample_documents": sample_info
        }
        
    except Exception as e:
        return {"error": str(e)}

# 6. Fix for startup using lifespan context manager
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    # Startup
    global supabase_client, hf_client
    
    logger.info("🚀 Starting InfinitePay AI Chatbot v1.2.0")
    
    try:
        await initialize_services()
        yield
    finally:
        # Cleanup
        logger.info("🔄 Shutting down services...")
        if supabase_client:
            # Add any cleanup needed for supabase
            supabase_client = None
        if hf_client:
            # Add any cleanup needed for hugging face
            hf_client = None

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

async def initialize_services():
    """Initialize all required services."""
    global supabase_client, hf_client

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
            
            # FIX 10: Test the connection immediately
            test_result = supabase_client.table('documents').select('id', count='exact').limit(1).execute()
            logger.info(f"✅ Supabase client initialized. Found {test_result.count} documents.")
            
            # Check for embeddings
            embedding_test = supabase_client.table('documents').select('embedding').limit(1).execute()
            if embedding_test.data and embedding_test.data[0].get('embedding'):
                embedding_dim = len(embedding_test.data[0]['embedding'])
                logger.info(f"📐 Database embedding dimension: {embedding_dim}")
            else:
                logger.warning("⚠️ No embeddings found in database!")
                
        except Exception as e:
            logger.error(f"❌ Supabase initialization failed: {e}")
            supabase_client = None
    else:
        logger.warning("⚠️ Supabase credentials not found. Database features will be disabled.")

    # Initialize Hugging Face Inference Client
    if hf_token:
        try:
            hf_client = InferenceClient(
                provider="auto",
                api_key=hf_token
            )
            
            # FIX 11: Test the HF client immediately
            test_embedding = await asyncio.to_thread(
                hf_client.feature_extraction,
                "test",
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            hf_embedding_dim = len(test_embedding.tolist() if hasattr(test_embedding, 'tolist') else test_embedding)
            logger.info(f"✅ Hugging Face client initialized. Embedding dimension: {hf_embedding_dim}")
            
        except Exception as e:
            logger.error(f"❌ Hugging Face Inference Client initialization failed: {e}")
            hf_client = None
    else:
        logger.warning("⚠️ HF_TOKEN not found. AI features will be disabled.")

    logger.info("🎉 Services initialization complete.")