import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client
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
        print("\n📄 Getting sample documents for filter 'planos'...")
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
    
# Declare global variables
supabase_ok = False

async def main():
    """Run all tests"""
    global supabase_ok  # Declare global variable

    print("🚀 Starting Supabase and Hugging Face tests...\n")
    
    # Test 1: Supabase connection
    print("=" * 50)
    print("TEST 1: Supabase Connection")
    print("=" * 50)
    supabase_ok = await test_supabase_connection()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Supabase Connection: {'✅ PASS' if supabase_ok else '❌ FAIL'}")
    
    if supabase_ok:  # Add proper indentation and check
        print("\n🎉 All tests passed! The setup should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())     