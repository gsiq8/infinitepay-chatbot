#!/usr/bin/env python3
"""
Fixed Supabase Upload Script for Secure Setup
Uploads processed data to Supabase with proper RLS handling
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_supabase_connection():
    """Test basic Supabase connection with both keys"""
    try:
        from supabase import create_client
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_anon_key = os.getenv('SUPABASE_ANON_KEY')
        supabase_service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not supabase_url:
            logger.error("Missing SUPABASE_URL in .env file")
            return False
            
        if not supabase_anon_key:
            logger.error("Missing SUPABASE_ANON_KEY in .env file")
            return False
        
        # Test with anon key (read-only access)
        logger.info("Testing connection with anon key...")
        supabase_anon = create_client(supabase_url, supabase_anon_key)
        
        try:
            result = supabase_anon.table('documents').select('id').limit(1).execute()
            logger.info("✅ Anon key connection successful")
        except Exception as e:
            if "relation \"public.documents\" does not exist" in str(e) or "404" in str(e):
                logger.error("❌ Documents table does not exist")
                logger.error("Please run the SQL script in Supabase Dashboard first")
                return False
            else:
                logger.warning(f"⚠️ Anon key test failed (this might be normal if table is empty): {e}")
        
        # Test with service role key if available
        if supabase_service_key:
            logger.info("Testing connection with service role key...")
            supabase_service = create_client(supabase_url, supabase_service_key)
            
            try:
                result = supabase_service.table('documents').select('id').limit(1).execute()
                logger.info("✅ Service role key connection successful")
                logger.info("✅ Ready for secure uploads")
                return True
            except Exception as e:
                if "relation \"public.documents\" does not exist" in str(e):
                    logger.error("❌ Documents table does not exist")
                    return False
                else:
                    logger.warning(f"⚠️ Service role test warning: {e}")
                    return True  # Might still work for uploads
        else:
            logger.warning("⚠️ SUPABASE_SERVICE_ROLE_KEY not found in .env")
            logger.warning("⚠️ Upload may fail due to RLS policies")
            logger.info("Add SUPABASE_SERVICE_ROLE_KEY to .env for secure uploads")
            return True  # Let's try anyway
            
    except ImportError:
        logger.error("Supabase package not installed. Run: pip install supabase")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing Supabase: {e}")
        return False

def upload_data():
    """Upload processed data to Supabase using service role"""
    logger.info("Starting data upload to Supabase...")
    
    # Check if processed data exists
    if not Path('processed_data/chunks.json').exists():
        logger.error("❌ No processed data found")
        logger.error("Please run: python quick_start_script.py")
        return False
    
    try:
        from supabase import create_client
        
        supabase_url = os.getenv('SUPABASE_URL')
        service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        anon_key = os.getenv('SUPABASE_ANON_KEY')
        
        # Prefer service role key for uploads (bypasses RLS)
        if service_role_key:
            logger.info("Using service role key for upload (bypasses RLS)...")
            supabase = create_client(supabase_url, service_role_key)
        elif anon_key:
            logger.info("Using anon key for upload (may fail due to RLS)...")
            supabase = create_client(supabase_url, anon_key)
        else:
            logger.error("❌ No valid Supabase key found in .env")
            return False
        
        # Load processed data
        logger.info("Loading processed data...")
        with open('processed_data/chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        embeddings = np.load('processed_data/embeddings.npy')
        
        logger.info(f"Loaded {len(chunks)} chunks with {embeddings.shape[1]}D embeddings")
        
        # Test upload with a single document first
        logger.info("Testing upload with single document...")
        try:
            test_doc = {
                'content': "Test document for upload verification",
                'metadata': {"test": True},
                'embedding': embeddings[0].tolist(),
                'page_title': "Test Page",
                'page_url': "test://example.com",
                'chunk_index': 0,
                'is_public': True,
                'user_id': None  # Will be handled by function default
            }
            
            test_result = supabase.table('documents').insert([test_doc]).execute()
            logger.info("✅ Test upload successful")
            
            # Delete test document
            if test_result.data:
                test_id = test_result.data[0]['id']
                supabase.table('documents').delete().eq('id', test_id).execute()
                logger.info("✅ Test document cleaned up")
                
        except Exception as e:
            logger.error(f"❌ Test upload failed: {e}")
            logger.error("This might be due to RLS policies. Check your Supabase key permissions.")
            return False
        
        # Upload in batches
        batch_size = 20  # Smaller batches for better reliability
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        logger.info(f"Uploading in {total_batches} batches of {batch_size}...")
        
        successful_uploads = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_num = i // batch_size + 1
            
            try:
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                documents = []
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    doc = {
                        'content': chunk['content'],
                        'metadata': chunk.get('metadata', {}),
                        'embedding': embedding.tolist(),
                        'page_title': chunk.get('page_title', ''),
                        'page_url': chunk.get('page_url', ''),
                        'chunk_index': chunk.get('chunk_index', 0),
                        'is_public': True,  # Make documents publicly readable
                        'user_id': None  # Will use function default
                    }
                    documents.append(doc)
                
                # Upload batch using the insert_document function for better RLS handling
                if service_role_key:
                    # Direct table insert (service role bypasses RLS)
                    result = supabase.table('documents').insert(documents).execute()
                else:
                    # Use the function for each document (respects RLS)
                    for doc in documents:
                        result = supabase.rpc('insert_document', {
                            'p_content': doc['content'],
                            'p_metadata': doc['metadata'],
                            'p_embedding': doc['embedding'],
                            'p_page_title': doc['page_title'],
                            'p_page_url': doc['page_url'],
                            'p_chunk_index': doc['chunk_index'],
                            'p_is_public': doc['is_public'],
                            'p_user_id': doc['user_id']
                        }).execute()
                
                successful_uploads += len(documents)
                logger.info(f"✅ Batch {batch_num}/{total_batches}: {len(documents)} documents uploaded")
                
            except Exception as e:
                logger.error(f"❌ Error uploading batch {batch_num}/{total_batches}: {e}")
                
                # Show detailed error for first few failures
                if batch_num <= 3:
                    logger.error(f"Detailed error: {str(e)}")
                    
                    # Check if it's an RLS error
                    if "policy" in str(e).lower() or "permission" in str(e).lower():
                        logger.error("💡 This looks like an RLS policy error")
                        logger.error("💡 Make sure you're using SUPABASE_SERVICE_ROLE_KEY")
                
                continue
        
        logger.info(f"🎉 Upload completed!")
        logger.info(f"✅ Successfully uploaded: {successful_uploads}/{len(chunks)} documents")
        
        if successful_uploads < len(chunks):
            logger.warning(f"⚠️ {len(chunks) - successful_uploads} documents failed to upload")
        
        return successful_uploads > 0
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return False

def verify_upload():
    """Verify the upload was successful"""
    logger.info("Verifying upload...")
    
    try:
        from supabase import create_client
        
        # Use anon key for verification (tests read access)
        supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_ANON_KEY')
        )
        
        # Count documents using the safe view
        try:
            result = supabase.from_('documents_safe').select('id', count='exact').execute()
            count = result.count
            logger.info(f"✅ Found {count} documents via safe view")
        except Exception as e:
            # Fallback to direct table query
            logger.info("Safe view failed, trying direct table query...")
            result = supabase.table('documents').select('id', count='exact').execute()
            count = result.count
            logger.info(f"✅ Found {count} documents via direct query")
        
        # Test similarity search function
        if count > 0:
            logger.info("Testing similarity search function...")
            
            # Get a sample embedding
            sample_result = supabase.table('documents').select('embedding').limit(1).execute()
            
            if sample_result.data:
                sample_embedding = sample_result.data[0]['embedding']
                
                # Test the similarity search function
                search_result = supabase.rpc(
                    'similarity_search',
                    {
                        'query_embedding': sample_embedding,
                        'match_count': 3
                    }
                ).execute()
                
                if search_result.data:
                    logger.info(f"✅ Similarity search working - found {len(search_result.data)} results")
                    
                    # Show sample result
                    sample = search_result.data[0]
                    logger.info(f"✅ Sample result similarity: {sample.get('similarity', 'N/A')}")
                    return True
                else:
                    logger.error("❌ Similarity search returned no results")
                    return False
            else:
                logger.error("❌ No documents found to test with")
                return False
        
        return count > 0
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        logger.error("This might be due to RLS policies preventing read access")
        return False

def main():
    """Main function"""
    logger.info("🚀 Secure Supabase Data Upload")
    
    # Test connection
    if not test_supabase_connection():
        logger.error("❌ Supabase connection failed")
        logger.info("\n📋 TROUBLESHOOTING:")
        logger.info("1. Check your .env file has correct SUPABASE_URL and SUPABASE_ANON_KEY")
        logger.info("2. Add SUPABASE_SERVICE_ROLE_KEY to .env for secure uploads")
        logger.info("3. Run the SQL script in Supabase Dashboard → SQL Editor")
        logger.info("4. Make sure the 'documents' table exists with RLS enabled")
        return False
    
    # Upload data
    if not upload_data():
        logger.error("❌ Data upload failed")
        return False
    
    # Verify upload
    if not verify_upload():
        logger.warning("⚠️ Upload verification had issues, but upload may still be successful")
        logger.info("Check your Supabase dashboard to verify the data")
    
    logger.info("🎉 Supabase setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. python run_chatbot.py  - Start the API server")
    logger.info("2. python api_test_script.py  - Test the complete system")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)