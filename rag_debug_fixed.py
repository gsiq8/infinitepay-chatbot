#!/usr/bin/env python3
"""
Debug script to identify issues with RAG system - FIXED VERSION
"""

import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client
from huggingface_hub import InferenceClient
import logging
import numpy as np

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
        
        # Step 1: Generate embedding (FIXED VERSION)
        try:
            print("1️⃣ Generating embedding...")
            embedding_result = await asyncio.to_thread(
                hf_client.feature_extraction,
                query,
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Apply the same fixing logic as main.py
            if embedding_result is None:
                print("   ❌ Received None from Hugging Face API")
                continue
            
            # Convert to numpy array for easier manipulation
            embedding_array = np.array(embedding_result)
            print(f"   📐 Raw embedding shape: {embedding_array.shape}")
            
            # Handle different possible shapes:
            if embedding_array.ndim == 1:
                # Already 1D, perfect
                final_embedding = embedding_array.tolist()
                print(f"   ✅ Using 1D embedding, length: {len(final_embedding)}")
            elif embedding_array.ndim == 2:
                if embedding_array.shape[0] == 1:
                    # Single row, flatten it
                    final_embedding = embedding_array.flatten().tolist()
                    print(f"   ✅ Flattened 2D embedding from shape {embedding_array.shape} to length: {len(final_embedding)}")
                else:
                    # Multiple rows, use mean pooling
                    final_embedding = np.mean(embedding_array, axis=0).tolist()
                    print(f"   ✅ Mean pooled 2D embedding from shape {embedding_array.shape} to length: {len(final_embedding)}")
            else:
                print(f"   ❌ Unexpected embedding shape: {embedding_array.shape}")
                continue
                
        except Exception as e:
            print(f"   ❌ Embedding failed: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            continue
        
        # Step 2: Perform similarity search
        try:
            print("2️⃣ Performing similarity search...")
            result = supabase_client.rpc(
                'similarity_search',
                {'query_embedding': final_embedding, 'match_count': 5}
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
        # Generate embedding (FIXED VERSION)
        embedding_result = await asyncio.to_thread(
            hf_client.feature_extraction,
            test_query,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Apply the same fixing logic
        embedding_array = np.array(embedding_result)
        
        if embedding_array.ndim == 1:
            final_embedding = embedding_array.tolist()
        elif embedding_array.ndim == 2:
            if embedding_array.shape[0] == 1:
                final_embedding = embedding_array.flatten().tolist()
            else:
                final_embedding = np.mean(embedding_array, axis=0).tolist()
        else:
            print(f"❌ Unexpected embedding shape: {embedding_array.shape}")
            return
        
        print(f"🔍 Testing with query: '{test_query}'")
        print(f"📐 Embedding dimensions: {len(final_embedding)}")
        print(f"📐 Original shape: {embedding_array.shape}")
        
        # Try with different match_count values
        for match_count in [3, 5, 10]:
            print(f"\n🎯 Testing with match_count={match_count}")
            
            result = supabase_client.rpc(
                'similarity_search',
                {'query_embedding': final_embedding, 'match_count': match_count}
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
        import traceback
        print(f"Traceback: {traceback.format_exc()}")


async def check_main_py_logic():
    """Simulate the exact logic from main.py - FIXED VERSION"""
    
    print("\n" + "=" * 50)
    print("MAIN.PY LOGIC SIMULATION - FIXED")
    print("=" * 50)
    
    # Copy the exact RAGService logic WITH THE FIX
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
                # Use the sentence-transformers model for embeddings
                embedding_result = await asyncio.to_thread(
                    hf_client.feature_extraction,
                    text,
                    model="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Handle different response formats from HF API
                if embedding_result is None:
                    logger.error("❌ Received None from Hugging Face API")
                    return None
                
                # Convert to numpy array for easier manipulation
                embedding_array = np.array(embedding_result)
                logger.info(f"📐 Raw embedding shape: {embedding_array.shape}")
                
                # Handle different possible shapes:
                if embedding_array.ndim == 1:
                    # Already 1D, perfect
                    final_embedding = embedding_array.tolist()
                    logger.info(f"✅ Using 1D embedding, length: {len(final_embedding)}")
                elif embedding_array.ndim == 2:
                    if embedding_array.shape[0] == 1:
                        # Single row, flatten it
                        final_embedding = embedding_array.flatten().tolist()
                        logger.info(f"✅ Flattened 2D embedding from shape {embedding_array.shape} to length: {len(final_embedding)}")
                    else:
                        # Multiple rows, use mean pooling
                        final_embedding = np.mean(embedding_array, axis=0).tolist()
                        logger.info(f"✅ Mean pooled 2D embedding from shape {embedding_array.shape} to length: {len(final_embedding)}")
                else:
                    logger.error(f"❌ Unexpected embedding shape: {embedding_array.shape}")
                    return None
                
                # Validate final embedding
                if not final_embedding or not all(isinstance(x, (int, float)) for x in final_embedding):
                    logger.error("❌ Invalid embedding format after processing")
                    return None
                    
                return final_embedding
                
            except Exception as e:
                logger.error(f"❌ Error getting embedding from Hugging Face: {e}")
                logger.error(f"   Error type: {type(e)}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
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


async def test_embedding_shapes():
    """Test what shapes we get from HF API"""
    
    print("\n" + "=" * 50)
    print("EMBEDDING SHAPES ANALYSIS")
    print("=" * 50)
    
    hf_token = os.getenv('HF_TOKEN')
    hf_client = InferenceClient(
        provider="auto",
        api_key=hf_token
    )
    
    test_texts = [
        "Como funciona?",
        "InfinitePay maquininha de cartão",
        "Qual é a taxa cobrada para pagamentos?",
        "Pix"
    ]
    
    for text in test_texts:
        try:
            print(f"\n🔍 Testing: '{text}'")
            
            embedding_result = await asyncio.to_thread(
                hf_client.feature_extraction,
                text,
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            print(f"   Type: {type(embedding_result)}")
            
            if isinstance(embedding_result, (list, tuple)):
                print(f"   Length: {len(embedding_result)}")
                if len(embedding_result) > 0:
                    print(f"   First element type: {type(embedding_result[0])}")
                    if isinstance(embedding_result[0], (list, tuple)):
                        print(f"   First element length: {len(embedding_result[0])}")
            
            # Convert to numpy to check shape
            arr = np.array(embedding_result)
            print(f"   Numpy shape: {arr.shape}")
            print(f"   Numpy dtype: {arr.dtype}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


async def main():
    """Run all debug tests"""
    print("🚀 RAG DEBUG SESSION STARTED - FIXED VERSION")
    print("=" * 60)
    
    try:
        await test_embedding_shapes()
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
        print("5. ✅ Simulate exact main.py logic (FIXED)")
        print("6. ✅ Analyze embedding shapes from HF API")
        
    except Exception as e:
        print(f"❌ Debug session failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())