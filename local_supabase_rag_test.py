import os
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalSupabaseRAGTester:
    def __init__(self, 
                 supabase_url: str, 
                 supabase_key: str,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 table_name: str = "documents"):
        
        # Initialize Supabase client
        self.supabase: Client = None
        self.table_name = table_name
        
        # Initialize local embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Ollama settings
        self.ollama_base_url = "http://localhost:11434"
        
    def test_supabase_connection(self):
        """Test connection to Supabase"""
        try:
            result = self.supabase.table(self.table_name).select("*").limit(1).execute()
            logger.info(f"✅ Supabase connection successful! Found {len(result.data)} records")
            return True
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")
            return False
    
    def test_ollama_connection(self, model_name: str = "llama3.2:latest"):
        """Test connection to local Ollama"""
        try:
            url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": model_name,
                "prompt": "Hello, respond with just 'OK'",
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            logger.info(f"✅ Ollama connection successful with {model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            return False
    
    def get_documents_from_supabase(self, limit: int = None) -> List[Dict]:
        """Retrieve documents from Supabase"""
        try:
            query = self.supabase.table(self.table_name).select("*")
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            logger.info(f"📚 Retrieved {len(result.data)} documents from Supabase")
            return result.data
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_embeddings_locally(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local model"""
        logger.info(f"🧠 Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def update_supabase_embeddings(self, documents: List[Dict], content_field: str = "content"):
        """Update Supabase with locally generated embeddings"""
        logger.info("📤 Updating Supabase with local embeddings...")
        
        # Extract text content
        texts = [doc.get(content_field, "") for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_locally(texts)
        
        # Update each document
        for i, doc in enumerate(documents):
            try:
                # Convert numpy array to list for JSON serialization
                embedding_list = embeddings[i].tolist()
                
                update_data = {"embedding": embedding_list}
                
                result = self.supabase.table(self.table_name)\
                    .update(update_data)\
                    .eq("id", doc["id"])\
                    .execute()
                
                if i % 10 == 0:  # Progress update
                    logger.info(f"Updated {i+1}/{len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error updating document {doc.get('id')}: {e}")
        
        logger.info("✅ Finished updating embeddings in Supabase")
    
    def similarity_search_local(self, 
                               query: str, 
                               top_k: int = 5, 
                               content_field: str = "content") -> List[Dict]:
        """Perform similarity search using local embeddings"""
        logger.info(f"🔍 Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Get all documents with embeddings from Supabase
        try:
            result = self.supabase.table(self.table_name)\
                .select(f"*, {content_field}")\
                .not_.is_("embedding", "null")\
                .execute()
            
            documents = result.data
            
            if not documents:
                logger.warning("No documents with embeddings found")
                return []
            
            # Calculate similarities
            similarities = []
            for doc in documents:
                if doc.get("embedding"):
                    doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
                    similarity = np.dot(query_embedding, doc_embedding.T)[0][0]
                    similarities.append({
                        "document": doc,
                        "similarity": similarity
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Return top k results
            top_results = similarities[:top_k]
            
            logger.info(f"📊 Found {len(top_results)} similar documents:")
            for i, result in enumerate(top_results):
                score = result["similarity"]
                content = result["document"].get(content_field, "")[:100]
                logger.info(f"  {i+1}. Score: {score:.4f} - {content}...")
            
            return top_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def chat_with_ollama(self, model: str, prompt: str, context: str = "") -> str:
        """Generate response using local Ollama"""
        logger.info(f"💬 Generating response with {model}")
        
        full_prompt = f"""Contexto: {context}

Pergunta: {prompt}

Responda com base no contexto fornecido em português:"""
        
        try:
            url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["response"]
            
        except Exception as e:
            logger.error(f"Error with Ollama: {e}")
            return f"Erro ao conectar com o modelo: {e}"
    
    def full_rag_pipeline(self, 
                         query: str, 
                         model: str = "llama3.2:latest",
                         top_k: int = 3,
                         content_field: str = "content") -> Dict[str, Any]:
        """Complete RAG pipeline: search + generate"""
        logger.info(f"🚀 Full RAG pipeline for: '{query}'")
        logger.info("-" * 60)
        
        # 1. Similarity search
        search_results = self.similarity_search_local(query, top_k, content_field)
        
        if not search_results:
            return {
                "query": query,
                "error": "No similar documents found",
                "search_results": [],
                "response": "Desculpe, não encontrei informações relevantes para sua pergunta."
            }
        
        # 2. Prepare context
        context_parts = []
        for result in search_results:
            content = result["document"].get(content_field, "")
            context_parts.append(content)
        
        context = "\n\n".join(context_parts)
        
        # 3. Generate response
        response = self.chat_with_ollama(model, query, context)
        
        logger.info(f"📝 Generated response: {response[:200]}...")
        
        return {
            "query": query,
            "search_results": search_results,
            "context": context,
            "response": response
        }

def main():
    # Configuration - Replace with your actual values
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
    
    # Initialize tester
    tester = LocalSupabaseRAGTester(
        supabase_url=SUPABASE_URL,
        supabase_key=SUPABASE_KEY,
    )
    
    # Test connections
    print("🧪 Testing Connections")
    print("=" * 50)
    
    supabase_ok = tester.test_supabase_connection()
    ollama_ok = tester.test_ollama_connection("llama3.2:latest")
    
    if not supabase_ok or not ollama_ok:
        print("❌ Some connections failed. Please check your setup.")
        return
    
    # Get documents
    print("\n📚 Loading Documents")
    print("=" * 50)
    documents = tester.get_documents_from_supabase(limit=50)  # Test with first 50
    
    if not documents:
        print("❌ No documents found in Supabase")
        return
    
    # Check if embeddings exist
    has_embeddings = any(doc.get("embedding") for doc in documents)
    
    if not has_embeddings:
        print("🧠 No embeddings found. Generating locally...")
        tester.update_supabase_embeddings(documents, CONTENT_FIELD)
    else:
        print("✅ Embeddings found in database")
    
    # Test queries
    print("\n🔍 Testing RAG Pipeline")
    print("=" * 50)
    
    test_queries = [
        "Quais os planos disponíveis?",
        "Como funciona o cancelamento?",
        "Quais são as taxas?",
        "Horário de atendimento"
    ]
    
    models_to_test = ["llama3.2:latest", "gemma:2b"]
    
    for model in models_to_test:
        if not tester.test_ollama_connection(model):
            continue
            
        print(f"\n🤖 Testing with {model}")
        print("-" * 40)
        
        for query in test_queries[:2]:  # Test first 2 queries
            try:
                result = tester.full_rag_pipeline(
                    query=query,
                    model=model,
                    top_k=3,
                    content_field=CONTENT_FIELD
                )
                
                print(f"\n📋 Results for: '{query}'")
                print(f"Response: {result['response']}")
                print("\n" + "="*60 + "\n")
                
            except Exception as e:
                logger.error(f"Error testing {model} with query '{query}': {e}")

if __name__ == "__main__":
    main()