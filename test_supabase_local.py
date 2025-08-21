#!/usr/bin/env python3
"""
Test script to verify Supabase connection and Hugging Face Inference
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

async def test_supabase_connection():
    """Test basic Supabase connection and data access"""
    
    # Get environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    hf_token = os.getenv('HF_TOKEN')
    
    print("🔍 Environment check:")
    print(f"   SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Not set'}")
    print(f"   SUPABASE_KEY: {'✅ Set' if supabase_key else '❌ Not set'}")
    print(f"   HF_TOKEN: {'✅ Set' if hf_token else '❌ Not set'}")
    
    if not supabase_url or not supabase_key:
        print("❌ Supabase credentials not found!")
        return False
    
    try:
        # Initialize Supabase client
        supabase_client = create_client(supabase_url, supabase_key)
        print("✅ Supabase client initialized successfully")
        
        # Test basic table access
        print("\n📊 Testing basic table access...")
        result = supabase_client.table('documents').select('id', count='exact').limit(1).execute()
        print(f"   Basic query result: {result}")
        
        # Get total document count
        count_result = supabase_client.table('documents').select('*', count='exact').execute()
        print(f"   Total documents in table: {count_result.count}")
        
        # Get a few sample documents with more details
        print("\n📄 Getting sample documents...")
        sample_docs = supabase_client.table('documents').select('id, page_title, content, page_url').ilike('content', '%planos%').limit(3).execute()
        print(f"   Sample documents: {len(sample_docs.data)}")
        for i, doc in enumerate(sample_docs.data):
            title = doc.get('page_title', 'No title')
            content_preview = doc.get('content', '')[:100] if doc.get('content') else 'No content'
            print(f"   Doc {i+1}: {title}")
            print(f"      Content preview: {content_preview}...")
            print(f"      URL: {doc.get('page_url', 'No URL')}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        return False

async def test_hf_chat():
    """Test Hugging Face chat completion"""
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found!")
        return False
    
    try:
        # Initialize Hugging Face client with correct syntax
        hf_client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token,
        )
        
        completion = hf_client.chat.completions.create(
            
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
        )
        
        print(completion.choices[0].message)
        
        return True
        
    except Exception as e:
        print(f"❌ Error with Hugging Face chat: {e}")
        return False


async def test_hf_sentence_similarity():
    """Test Hugging Face sentence similarity"""
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found!")
        return False
    
    try:
        # Initialize Hugging Face client with correct syntax
        hf_client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token,
        )

        # Test sentence similarity with correct syntax
        test_text = "Quanto é a maquininha?",
        other_sentences = [
                    "InfinitePay é uma empresa de pagamentos",
                    "Como funciona o pagamento",
                    "Serviços financeiros"
                ]
        print(f"\n🧠 Testing sentence similarity for: '{test_text, other_sentences}'")


        result = hf_client.sentence_similarity(
                sentence=test_text,
                other_sentences=other_sentences,
                model="sentence-transformers/all-MiniLM-L6-v2",
        )

        print(f"✅ Sentence similarity successful (separate params): {result}")

        return True
        
    except Exception as e1:
        print(f"⚠️  Separate parameters failed: {e1}")
        return False

async def test_similarity_search():
    """Test similarity search with Supabase"""
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    hf_token = os.getenv('HF_TOKEN')
    
    if not all([supabase_url, supabase_key, hf_token]):
        print("❌ Missing required environment variables!")
        print(f"   SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Not set'}")
        print(f"   SUPABASE_KEY: {'✅ Set' if supabase_key else '❌ Not set'}")
        print(f"   HF_TOKEN: {'✅ Set' if hf_token else '❌ Not set'}")
        return False
    
    try:
        # Initialize clients
        supabase_client = create_client(supabase_url, supabase_key)
        hf_client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token
        )
        
        # Test query
        test_query = "Quais são os "
        print(f"\n🔍 Testing similarity search for: '{test_query}'")
        
        # Generate embedding using the correct API
        print("🔍 Generating embedding...")
        embedding = hf_client.feature_extraction(
            test_query,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        print(f"✅ Query embedding generated, type: {type(embedding)}")
        
        # Convert to list (works for numpy arrays and other array-like objects)
        embedding_list = embedding.tolist()
        print(f"✅ Embedding converted to list, length: {len(embedding_list)}")
        
        # Test similarity search
        print("🔍 Performing similarity search...")
        result = supabase_client.rpc(
            'similarity_search',
            {'query_embedding': embedding_list, 'match_count': 5}
        ).execute()
        
        print(f"✅ Similarity search completed")
        print(f"   Results found: {len(result.data or [])}")
        
        if result.data:
            print("   Top results:")
            for i, doc in enumerate(result.data[:3]):
                similarity = doc.get('similarity', 'N/A')
                if isinstance(similarity, (int, float)):
                    similarity = f"{similarity:.4f}"
                print(f"   {i+1}. {doc.get('page_title', 'No title')[:50]}... (similarity: {similarity})")
        else:
            print("   No results found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Supabase and Hugging Face tests...\n")
    
    # Test 1: Supabase connection
    print("=" * 50)
    print("TEST 1: Supabase Connection")
    print("=" * 50)
    supabase_ok = await test_supabase_connection()
    
    # Test 2: Hugging Face Chat
    print("\n" + "=" * 50)
    print("TEST 2: Hugging Face Chat")
    print("=" * 50)
    chat_ok = await test_hf_chat()
    
    # Test 3: Hugging Face Sentence Similarity
    print("\n" + "=" * 50)
    print("TEST 4: Hugging Face Sentence Similarity")
    print("=" * 50)
    similarity_ok = await test_hf_sentence_similarity()
    
    # Test 4: Similarity search (only if previous tests pass)
    if all([supabase_ok, similarity_ok]):
        print("\n" + "=" * 50)
        print("TEST 5: Similarity Search")
        print("=" * 50)
        search_ok = await test_similarity_search()
    else:
        search_ok = False
        print("\n" + "=" * 50)
        print("TEST 5: Similarity Search - SKIPPED (prerequisites not met)")
        print("=" * 50)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Supabase Connection: {'✅ PASS' if supabase_ok else '❌ FAIL'}")
    print(f"Hugging Face Chat: {'✅ PASS' if chat_ok else '❌ FAIL'}")
    print(f"Hugging Face Similarity: {'✅ PASS' if similarity_ok else '❌ FAIL'}")
    print(f"Similarity Search: {'✅ PASS' if search_ok else '❌ FAIL'}")
    
    if all([supabase_ok, chat_ok, similarity_ok, search_ok]):
        print("\n🎉 All tests passed! The setup should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main()) 