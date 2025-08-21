#!/usr/bin/env python3
"""
Debug script to identify issues with RAG system
"""

import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client
from huggingface_hub import InferenceClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def debug_rag_pipeline():
    """Debug the entire RAG pipeline step by step"""
    
    # Get environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    hf_token = os.getenv('HF_TOKEN')
    
    print("🔍 RAG PIPELINE DEBUG")
    print("=" * 50)
    
    # Initialize clients
    supabase_client = create_client(supabase_url, supabase_key)
    hf_client = InferenceClient(
        provider="auto",
        api_key=hf_token
    )
    
    # Test queries that should return results
    test_queries = [
        "Como funciona o pagamento?",
        "InfinitePay",
        "maquininha",
        "cartão de crédito",
        "Pix",
        "boleto",
        "taxa",
        "como cobrar"
    ]
    
    for query in test_queries:
        print(f"\n🧪 Testing query: '{query}'")
        print("-" * 40)
        
        # Step 1: Generate embedding
        try:
            print("1️⃣ Generating embedding...")
            embedding = await asyncio.to_thread(
                hf_client.feature_extraction,
                query,
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            print(f"   ✅ Embedding generated: length={len(embedding)}")
        except Exception as e:
            print(f"   ❌ Embedding failed: {e}")
            continue
        
        # Step 2: Perform similarity search
        try:
            print("2️⃣ Performing similarity search...")
            result = supabase_client.rpc(
                'similarity_search',
                {'query_embedding': embedding.tolist(), 'match_count': 5}
            ).execute()
            
            num_results = len(result.data or [])
            print(f"   ✅ Search completed: {num_results} results")
            
            if result.data:
                for i, doc in enumerate(result.data[:3]):
                    similarity = doc.get('similarity', 0)
                    title = doc.get('page_title', 'No title')[:50]
                    content = doc.get('content', '')[:100]
                    print(f"   📄 Result {i+1}: similarity={similarity:.4f}")
                    print(f"      Title: {title}")
                    print(f"      Content: {content}...")
                    print()
                
                # Check if any results meet the threshold
                best_similarity = max(doc.get('similarity', 0) for doc in result.data)
                if best_similarity < 0.3:  # Adjust threshold as needed
                    print(f"   ⚠️  LOW SIMILARITY: Best score {best_similarity:.4f} < 0.3")
                else:
                    print(f"   ✅ GOOD SIMILARITY: Best score {best_similarity:.4f}")
            else:
                print("   ❌ No results returned")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            
        # Step 3: Check what the main.py logic would do
        print("3️⃣ Main.py logic check...")
        if not result.data:
            print("   ❌ Would return 'não encontrei informações relevantes'")
        else:
            print("   ✅ Would proceed to generate response")


async def check_database_content():
    """Check what's actually in the database"""
    
    print("\n" + "=" * 50)
    print("DATABASE CONTENT ANALYSIS")
    print("=" * 50)
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    supabase_client = create_client(supabase_url, supabase_key)
    
    try:
        # Get total count
        result = supabase_client.table('documents').select('*', count='exact').execute()
        print(f"📊 Total documents: {result.count}")
        
        # Sample some documents
        print("\n📄 Sample documents:")
        sample = supabase_client.table('documents').select('id, page_title, content, embedding').limit(10).execute()
        
        for i, doc in enumerate(sample.data):
            title = doc.get('page_title', 'No title') or 'No title'
            content = doc.get('content', '')[:150]
            has_embedding = doc.get('embedding') is not None
            embedding_length = len(doc.get('embedding', [])) if has_embedding else 0
            
            print(f"   Doc {i+1}: {title}")
            print(f"      Content: {content}...")
            print(f"      Embedding: {'✅' if has_embedding else '❌'} ({embedding_length} dims)")
            print()
            
        # Check for empty embeddings
        empty_embeddings = supabase_client.table('documents').select('id', count='exact').is_('embedding', 'null').execute()
        print(f"⚠️  Documents with null embeddings: {empty_embeddings.count}")
        
        # Check embedding dimensions
        if sample.data and sample.data[0].get('embedding'):
            first_embedding = sample.data[0]['embedding']
            print(f"🔢 Embedding dimension: {len(first_embedding)}")
            
    except Exception as e:
        print(f"❌ Database analysis failed: {e}")


async def test_similarity_function():
    """Test the similarity search function directly"""
    
    print("\n" + "=" * 50)
    print("SIMILARITY FUNCTION TEST")
    print("=" * 50)
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    hf_token = os.getenv('HF_TOKEN')
    
    supabase_client = create_client(supabase_url, supabase_key)
    hf_client = InferenceClient(
        provider="auto",
        api_key=hf_token
    )
    
    test_query = "Como funciona o pagamento InfinitePay?"
    
    try:
        # Generate embedding
        embedding = await asyncio.to_thread(
            hf_client.feature_extraction,
            test_query,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print(f"🔍 Testing with query: '{test_query}'")
        print(f"📐 Embedding dimensions: {len(embedding)}")
        
        # Try with different match_count values
        for match_count in [3, 5, 10]:
            print(f"\n🎯 Testing with match_count={match_count}")
            
            result = supabase_client.rpc(
                'similarity_search',
                {'query_embedding': embedding.tolist(), 'match_count': match_count}
            ).execute()
            
            if result.data:
                print(f"   ✅ Found {len(result.data)} results")
                best_score = max(doc.get('similarity', 0) for doc in result.data)
                worst_score = min(doc.get('similarity', 0) for doc in result.data)
                print(f"   📊 Similarity range: {worst_score:.4f} - {best_score:.4f}")
            else:
                print("   ❌ No results")
                
    except Exception as e:
        print(f"❌ Similarity function test failed: {e}")


async def check_main_py_logic():
    """Simulate the exact logic from main.py"""
    
    print("\n" + "=" * 50)
    print("MAIN.PY LOGIC SIMULATION")
    print("=" * 50)
    
    # Copy the exact RAGService logic
    class RAGService:
        @staticmethod
        async def get_embedding(text: str):
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            hf_token = os.getenv('HF_TOKEN')
            
            hf_client = InferenceClient(
                provider="auto",
                api_key=hf_token
            )
            
            try:
                logger.info(f"Getting embedding for text: {text[:50]}...")
                embedding = await asyncio.to_thread(
                    hf_client.feature_extraction,
                    text,
                    model="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info(f"✅ Embedding generated successfully, length: {len(embedding) if embedding else 0}")
                return embedding
            except Exception as e:
                logger.error(f"❌ Error getting embedding from Hugging Face: {e}")
                return None

        @staticmethod
        async def similarity_search(query: str, k: int = 5):
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            supabase_client = create_client(supabase_url, supabase_key)

            query_embedding = await RAGService.get_embedding(query)
            if not query_embedding:
                logger.error("Could not generate query embedding. Aborting search.")
                return []

            try:
                logger.info(f"Performing similarity search for query: '{query}' with embedding length: {len(query_embedding)}")
                result = supabase_client.rpc(
                    'similarity_search',
                    {'query_embedding': query_embedding, 'match_count': k}
                ).execute()
                logger.info(f"✅ Similarity search returned {len(result.data or [])} results.")
                logger.info(f"Raw result: {result}")
                return result.data or []
            except Exception as e:
                logger.error(f"❌ Error in similarity search RPC call: {e}")
                return []
    
    # Test with various queries
    test_queries = [
        "Como funciona o pagamento?",
        "InfinitePay maquininha",
        "taxa cartão crédito",
        "como usar Pix"
    ]
    
    for query in test_queries:
        print(f"\n🧪 Testing main.py logic with: '{query}'")
        
        relevant_docs = await RAGService.similarity_search(query)
        
        if not relevant_docs:
            print("   ❌ MAIN.PY WOULD RETURN: 'não encontrei informações relevantes'")
        else:
            print(f"   ✅ MAIN.PY WOULD PROCEED: Found {len(relevant_docs)} docs")
            for i, doc in enumerate(relevant_docs[:3]):
                similarity = doc.get('similarity', 'N/A')
                title = doc.get('page_title', 'No title')[:50]
                print(f"      Doc {i+1}: {title} (sim: {similarity})")


async def main():
    """Run all debug tests"""
    print("🚀 RAG DEBUG SESSION STARTED")
    print("=" * 60)
    
    try:
        await debug_rag_pipeline()
        await check_database_content()
        await test_similarity_function()
        await check_main_py_logic()
        
        print("\n" + "=" * 60)
        print("🎯 DEBUGGING COMPLETE")
        print("=" * 60)
        
        print("\n📋 CHECKLIST:")
        print("1. ✅ Check if embeddings are being generated correctly")
        print("2. ✅ Check if database contains documents with embeddings")
        print("3. ✅ Check similarity scores and thresholds")
        print("4. ✅ Check if similarity_search function is working")
        print("5. ✅ Simulate exact main.py logic")
        
    except Exception as e:
        print(f"❌ Debug session failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())