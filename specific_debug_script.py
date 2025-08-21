#!/usr/bin/env python3
"""
Specific debug script to identify why similarity search is not working
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

async def test_specific_queries():
    """Test the specific queries that are failing"""
    
    print("🔍 TESTING SPECIFIC FAILING QUERIES")
    print("=" * 60)
    
    # Initialize clients
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    hf_token = os.getenv('HF_TOKEN')
    
    supabase_client = create_client(supabase_url, supabase_key)
    hf_client = InferenceClient(provider="auto", api_key=hf_token)
    
    # Test the exact queries that are failing
    failing_queries = [
        "taxas",
        "Quanto são as taxas?", 
        "Quais os planos?",
        "Me conte sobre a InfinitePay"
    ]
    
    for query in failing_queries:
        print(f"\n{'='*20} TESTING: '{query}' {'='*20}")
        
        try:
            # Step 1: Generate embedding
            print("1️⃣ Generating embedding...")
            embedding_result = await asyncio.to_thread(
                hf_client.feature_extraction,
                query,
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Process embedding (same logic as main.py)
            if embedding_result is None:
                print("   ❌ Received None from Hugging Face API")
                continue
            
            embedding_array = np.array(embedding_result)
            print(f"   📐 Raw embedding shape: {embedding_array.shape}")
            
            if embedding_array.ndim == 1:
                final_embedding = embedding_array.tolist()
            elif embedding_array.ndim == 2:
                if embedding_array.shape[0] == 1:
                    final_embedding = embedding_array.flatten().tolist()
                else:
                    final_embedding = np.mean(embedding_array, axis=0).tolist()
            else:
                print(f"   ❌ Unexpected embedding shape: {embedding_array.shape}")
                continue
                
            print(f"   ✅ Final embedding length: {len(final_embedding)}")
            
            # Step 2: Test similarity search with different match_count
            for match_count in [3, 5, 10, 20]:
                print(f"\n2️⃣ Similarity search (match_count={match_count})...")
                result = supabase_client.rpc(
                    'similarity_search',
                    {'query_embedding': final_embedding, 'match_count': match_count}
                ).execute()
                
                if result.data:
                    print(f"   ✅ Found {len(result.data)} results")
                    
                    # Show top results with details
                    for i, doc in enumerate(result.data[:3]):
                        similarity = doc.get('similarity', 0)
                        title = doc.get('page_title', 'No title')
                        content = doc.get('content', '')[:200]
                        
                        print(f"   📄 Result {i+1}:")
                        print(f"      Similarity: {similarity:.4f}")
                        print(f"      Title: {title}")
                        print(f"      Content: {content}...")
                        print(f"      Full content length: {len(doc.get('content', ''))}")
                        print()
                else:
                    print(f"   ❌ No results for match_count={match_count}")
            
            # Step 3: Test different threshold values
            print("\n3️⃣ Testing different similarity thresholds...")
            if result.data:
                similarities = [doc.get('similarity', 0) for doc in result.data]
                max_sim = max(similarities)
                min_sim = min(similarities)
                avg_sim = sum(similarities) / len(similarities)
                
                print(f"   📊 Similarity stats:")
                print(f"      Max: {max_sim:.4f}")
                print(f"      Min: {min_sim:.4f}")
                print(f"      Avg: {avg_sim:.4f}")
                
                # Test different thresholds
                for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    above_threshold = [s for s in similarities if s >= threshold]
                    print(f"      Above {threshold}: {len(above_threshold)} docs")
            
        except Exception as e:
            print(f"❌ Error testing '{query}': {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")


async def analyze_database_for_keywords():
    """Look for documents that should match the queries"""
    
    print("\n" + "=" * 60)
    print("DATABASE KEYWORD ANALYSIS")
    print("=" * 60)
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    supabase_client = create_client(supabase_url, supabase_key)
    
    keywords_to_search = [
        "taxa",
        "taxas", 
        "plano",
        "planos",
        "InfinitePay",
        "infinitepay",
        "pagamento",
        "maquininha",
        "cartão"
    ]
    
    for keyword in keywords_to_search:
        print(f"\n🔍 Searching for documents containing '{keyword}'...")
        
        try:
            # Search in content
            result = supabase_client.table('documents').select('id, page_title, content').ilike('content', f'%{keyword}%').limit(5).execute()
            
            if result.data:
                print(f"   ✅ Found {len(result.data)} documents containing '{keyword}'")
                for i, doc in enumerate(result.data):
                    title = doc.get('page_title', 'No title') or 'No title'
                    content = doc.get('content', '')
                    
                    # Find where the keyword appears
                    keyword_pos = content.lower().find(keyword.lower())
                    if keyword_pos >= 0:
                        start = max(0, keyword_pos - 50)
                        end = min(len(content), keyword_pos + len(keyword) + 50)
                        context = content[start:end]
                        print(f"   📄 Doc {i+1} ({title}): ...{context}...")
                    else:
                        print(f"   📄 Doc {i+1} ({title}): {content[:100]}...")
            else:
                print(f"   ❌ No documents found containing '{keyword}'")
                
        except Exception as e:
            print(f"   ❌ Error searching for '{keyword}': {e}")


async def test_manual_similarity():
    """Manually test similarity between query and known good documents"""
    
    print("\n" + "=" * 60)
    print("MANUAL SIMILARITY TEST")
    print("=" * 60)
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    hf_token = os.getenv('HF_TOKEN')
    
    supabase_client = create_client(supabase_url, supabase_key)
    hf_client = InferenceClient(provider="auto", api_key=hf_token)
    
    # Get some documents that should match
    docs_result = supabase_client.table('documents').select('id, page_title, content, embedding').ilike('content', '%taxa%').limit(3).execute()
    
    if not docs_result.data:
        print("❌ No documents with 'taxa' found for manual testing")
        return
    
    query = "taxas"
    print(f"🔍 Manual similarity test for query: '{query}'")
    
    # Generate query embedding
    embedding_result = await asyncio.to_thread(
        hf_client.feature_extraction,
        query,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    embedding_array = np.array(embedding_result)
    if embedding_array.ndim == 2:
        query_embedding = embedding_array.flatten()
    else:
        query_embedding = embedding_array
    
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Calculate similarity with found documents
    for i, doc in enumerate(docs_result.data):
        if doc.get('embedding'):
            doc_embedding = np.array(doc['embedding'])
            print(f"\nDoc {i+1} embedding shape: {doc_embedding.shape}")
            
            # Calculate cosine similarity manually
            dot_product = np.dot(query_embedding, doc_embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_doc = np.linalg.norm(doc_embedding)
            similarity = dot_product / (norm_query * norm_doc)
            
            print(f"Manual cosine similarity: {similarity:.4f}")
            print(f"Document title: {doc.get('page_title', 'No title')}")
            print(f"Document content sample: {doc.get('content', '')[:200]}...")
        else:
            print(f"Doc {i+1}: No embedding found")


async def test_similarity_function_directly():
    """Test the Supabase similarity function directly"""
    
    print("\n" + "=" * 60)
    print("SUPABASE SIMILARITY FUNCTION TEST")
    print("=" * 60)
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    supabase_client = create_client(supabase_url, supabase_key)
    
    # Try to call the similarity_search function with a simple test
    try:
        # Create a dummy embedding vector (same dimension as stored embeddings)
        # First, check what dimension the stored embeddings have
        sample_doc = supabase_client.table('documents').select('embedding').limit(1).execute()
        
        if sample_doc.data and sample_doc.data[0].get('embedding'):
            embedding_dim = len(sample_doc.data[0]['embedding'])
            print(f"📐 Database embedding dimension: {embedding_dim}")
            
            # Create a test vector with the same dimension
            test_embedding = [0.1] * embedding_dim
            
            print("🧪 Testing similarity function with dummy vector...")
            result = supabase_client.rpc(
                'similarity_search',
                {'query_embedding': test_embedding, 'match_count': 5}
            ).execute()
            
            if result.data:
                print(f"✅ Similarity function works! Returned {len(result.data)} results")
                for i, doc in enumerate(result.data[:2]):
                    print(f"   Result {i+1}: similarity={doc.get('similarity', 'N/A')}")
            else:
                print("❌ Similarity function returned no results")
                
            # Check if there's an error in the result
            if hasattr(result, 'error') and result.error:
                print(f"❌ Similarity function error: {result.error}")
        else:
            print("❌ No sample document with embedding found")
            
    except Exception as e:
        print(f"❌ Error testing similarity function: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")


async def main():
    """Run specific debug tests for the failing queries"""
    
    print("🚨 SPECIFIC DEBUG FOR FAILING QUERIES")
    print("=" * 80)
    
    try:
        await test_specific_queries()
        await analyze_database_for_keywords()
        await test_manual_similarity()
        await test_similarity_function_directly()
        
        print("\n" + "=" * 80)
        print("🎯 SPECIFIC DEBUG COMPLETE")
        print("=" * 80)
        
        print("\n📋 DIAGNOSIS CHECKLIST:")
        print("1. ✅ Test embeddings for failing queries")
        print("2. ✅ Check if relevant documents exist in database")
        print("3. ✅ Test manual similarity calculations")
        print("4. ✅ Test Supabase similarity function")
        print("5. ✅ Analyze similarity thresholds")
        
    except Exception as e:
        print(f"❌ Debug session failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())