#!/usr/bin/env python3
"""
Quick Start Script for InfinitePay AI Chatbot
For when Ollama and models are already installed
Updated for secure Supabase setup
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

def check_environment():
    """Check environment setup"""
    logger.info("Checking environment...")
    
    # Check .env file
    if not Path('.env').exists():
        logger.warning(".env file not found, creating from example...")
        if Path('.env.example').exists():
            import shutil
            shutil.copy('.env.example', '.env')
            logger.info("Created .env file. Please edit it with your Supabase credentials.")
            return False
        else:
            logger.error("No .env.example file found")
            return False
    
    # Check required environment variables
    required_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        logger.error("Please set these in your .env file")
        return False
    
    # Check for service role key (recommended for uploads)
    if not os.getenv('SUPABASE_SERVICE_ROLE_KEY'):
        logger.warning("⚠️ SUPABASE_SERVICE_ROLE_KEY not found in .env")
        logger.warning("⚠️ This is recommended for secure uploads with RLS enabled")
        logger.info("Add it to your .env file for better upload reliability")
    
    logger.info("✅ Environment configured")
    return True

def check_ollama():
    """Verify Ollama is running with llama3.2"""
    logger.info("Checking Ollama setup...")
    
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            
            if response.status_code != 200:
                logger.error("❌ Ollama not responding")
                return False
            
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            # Check for llama3.2
            llama_models = [m for m in models if 'llama3.2' in m.lower()]
            
            if llama_models:
                logger.info(f"✅ Found llama3.2 model: {llama_models[0]}")
                return True
            else:
                logger.error("❌ llama3.2 model not found")
                logger.info(f"Available models: {models}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error checking Ollama: {e}")
        return False

def install_dependencies():
    """Install required Python packages"""
    logger.info("Installing dependencies...")
    
    packages = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "supabase==2.0.0",
        "sentence-transformers==2.2.2",
        "httpx==0.25.2",
        "python-multipart==0.0.6",
        "numpy==1.24.3",
        "python-dotenv==1.0.0"
    ]
    
    try:
        import subprocess
        for package in packages:
            logger.info(f"Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Failed to install {package}: {result.stderr}")
        
        logger.info("✅ Dependencies installed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error installing dependencies: {e}")
        return False

def process_data():
    """Process scraped data and create embeddings"""
    logger.info("Processing scraped data...")
    
    # Check if already processed
    if Path('processed_data/chunks.json').exists() and Path('processed_data/embeddings.npy').exists():
        logger.info("✅ Data already processed")
        return True
    
    # Check if scraped data exists
    if not Path('infinitepay_data/text_files').exists():
        logger.error("❌ No scraped data found in infinitepay_data/text_files/")
        logger.error("Please run the scraper first")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Initialize model
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Process files
        chunks = []
        texts = []
        
        text_files_dir = Path('infinitepay_data/text_files')
        
        for txt_file in text_files_dir.glob('*.txt'):
            logger.info(f"Processing {txt_file.name}...")
            
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if len(content) < 50:  # Skip very short files
                    continue
                
                # Simple chunking by sentences
                sentences = content.split('. ')
                chunk_size = 3  # 3 sentences per chunk
                
                for i in range(0, len(sentences), chunk_size):
                    chunk_sentences = sentences[i:i + chunk_size]
                    chunk_text = '. '.join(chunk_sentences)
                    
                    if len(chunk_text.strip()) > 50:
                        # Extract page title from filename
                        page_title = txt_file.stem.replace('page_', '').replace('_', ' ')
                        
                        chunk = {
                            'content': chunk_text,
                            'metadata': {
                                'source_file': txt_file.name,
                                'chunk_index': i // chunk_size
                            },
                            'page_title': page_title,
                            'page_url': f"https://infinitepay.io/{page_title.lower().replace(' ', '-')}",
                            'chunk_index': i // chunk_size
                        }
                        
                        chunks.append(chunk)
                        texts.append(chunk_text)
                        
            except Exception as e:
                logger.warning(f"Error processing {txt_file.name}: {e}")
        
        if not chunks:
            logger.error("❌ No valid chunks created")
            return False
        
        logger.info(f"Created {len(chunks)} chunks, generating embeddings...")
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Save processed data
        Path('processed_data').mkdir(exist_ok=True)
        
        with open('processed_data/chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        np.save('processed_data/embeddings.npy', embeddings)
        
        # Save summary
        summary = {
            'total_chunks': len(chunks),
            'embedding_dimensions': embeddings.shape[1],
            'model_used': 'all-MiniLM-L6-v2',
            'processing_date': str(Path('processed_data/chunks.json').stat().st_mtime)
        }
        
        with open('processed_data/processing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Data processed: {len(chunks)} chunks with {embeddings.shape[1]}D embeddings")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing data: {e}")
        return False

def print_supabase_setup():
    """Print secure Supabase setup instructions"""
    logger.info("Setting up Supabase with security...")
    
    sql_script = """-- Secure Supabase Setup for InfinitePay AI Chatbot
-- Fixes all security issues: RLS, search_path, extension schema, and dependency errors

-- 1. Create extensions schema (fix: Extension in Public)
CREATE SCHEMA IF NOT EXISTS extensions;

-- 2. Drop ALL dependent objects first (complete cleanup)
DROP VIEW IF EXISTS public.documents_safe;
DROP FUNCTION IF EXISTS public.similarity_search(vector, int);
DROP FUNCTION IF EXISTS public.insert_document(text, jsonb, vector, text, text, int);
DROP FUNCTION IF EXISTS public.insert_document(text, jsonb, vector, text, text, integer, boolean, uuid);
DROP INDEX IF EXISTS public.documents_embedding_idx;
DROP TABLE IF EXISTS public.documents;

-- 3. Now drop the vector extension from public schema
DROP EXTENSION IF EXISTS vector;

-- 4. Install the pgvector extension in extensions schema
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- 5. Create documents table with vector type (no schema prefix for type)
CREATE TABLE public.documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(384),  -- Vector type (extension is in extensions schema but type is still 'vector')
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    page_title TEXT,
    page_url TEXT,
    chunk_index INTEGER,
    is_public BOOLEAN DEFAULT true,  -- Added for RLS policy
    user_id UUID REFERENCES auth.users(id)  -- Added for user-specific access
);

-- 6. Enable Row Level Security (fix: RLS Disabled)
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- 7. Create RLS policies for the documents table
-- Policy for service role (full access)
CREATE POLICY "Enable full access for service role" ON public.documents
    FOR ALL USING (auth.role() = 'service_role');

-- Policy for anon role (read only for public documents)
CREATE POLICY "Enable read access for anon role" ON public.documents
    FOR SELECT USING (is_public = true);

-- Policy for authenticated users (read access based on ownership or public)
CREATE POLICY "Enable read access for authenticated users" ON public.documents
    FOR SELECT USING (
        is_public = true OR 
        user_id = auth.uid()
    );

-- 8. Create index with vector operators
CREATE INDEX documents_embedding_idx 
ON public.documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 9. Updated similarity search function
CREATE OR REPLACE FUNCTION public.similarity_search(
    query_embedding vector(384),
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id bigint,
    content text,
    metadata jsonb,
    page_title text,
    page_url text,
    similarity float
)
LANGUAGE sql 
STABLE
SECURITY DEFINER
SET search_path = public, extensions  -- Include extensions schema in search path
AS $$
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        documents.page_title,
        documents.page_url,
        1 - (documents.embedding <=> query_embedding) as similarity
    FROM public.documents
    WHERE (
        is_public = true OR 
        user_id = auth.uid() OR
        auth.role() = 'service_role'
    )
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- 10. Updated insert function
CREATE OR REPLACE FUNCTION public.insert_document(
    p_content text,
    p_metadata jsonb,
    p_embedding vector(384),
    p_page_title text DEFAULT NULL,
    p_page_url text DEFAULT NULL,
    p_chunk_index integer DEFAULT NULL,
    p_is_public boolean DEFAULT true,
    p_user_id uuid DEFAULT NULL
)
RETURNS bigint
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
    INSERT INTO public.documents (
        content, 
        metadata, 
        embedding, 
        page_title, 
        page_url, 
        chunk_index,
        is_public,
        user_id
    )
    VALUES (
        p_content, 
        p_metadata, 
        p_embedding, 
        p_page_title, 
        p_page_url, 
        p_chunk_index,
        p_is_public,
        COALESCE(p_user_id, auth.uid())
    )
    RETURNING id;
$$;

-- 11. Grant execute permissions on functions
GRANT EXECUTE ON FUNCTION public.similarity_search TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.insert_document TO authenticated, service_role;

-- 12. Create a regular view (fix: SECURITY DEFINER issue)
-- Using SECURITY INVOKER (default) instead of SECURITY DEFINER
CREATE OR REPLACE VIEW public.documents_safe 
WITH (security_invoker=on)
AS 
SELECT  
    id,
    content,
    metadata,
    page_title,
    page_url,
    chunk_index,
    created_at,
    is_public
FROM public.documents
WHERE (
    is_public = true OR 
    user_id = auth.uid() OR
    auth.role() = 'service_role'
);

-- 13. Grant permissions on the view
GRANT SELECT ON public.documents_safe TO anon, authenticated, service_role;

-- 14. Grant basic table permissions
GRANT SELECT ON public.documents TO anon, authenticated, service_role;
GRANT INSERT, UPDATE, DELETE ON public.documents TO authenticated, service_role;
GRANT USAGE ON SEQUENCE public.documents_id_seq TO authenticated, service_role;

-- 15. Verification and completion message
DO $$
BEGIN
    RAISE NOTICE 'Setup completed successfully!';
    RAISE NOTICE '✅ Extension moved to extensions schema';
    RAISE NOTICE '✅ RLS enabled on documents table with proper policies';  
    RAISE NOTICE '✅ Search path fixed in functions';
    RAISE NOTICE '✅ SECURITY DEFINER view issue resolved';
    RAISE NOTICE '✅ Dependency errors fixed by proper drop order';
    RAISE NOTICE '✅ Proper permissions granted';
    RAISE NOTICE '✅ Vector types properly referenced (no schema prefix needed)';
END $$;"""
    
    print("\n" + "="*60)
    print("📚 SECURE SUPABASE SETUP REQUIRED")
    print("="*60)
    print("Please run the following SQL in your Supabase SQL Editor:")
    print("(Dashboard > SQL Editor > New Query)")
    print("\n" + sql_script)
    print("="*60)
    print("\n⚠️  IMPORTANT: This script includes security fixes:")
    print("• Moves vector extension out of public schema")
    print("• Enables Row Level Security (RLS)")
    print("• Fixes SECURITY DEFINER view issues")
    print("• Proper permission management")
    
    return input("\nHave you run the SECURE SQL script in Supabase? (y/n): ").lower() == 'y'

def upload_to_supabase():
    """Upload processed data to Supabase with secure setup"""
    logger.info("Uploading data to secure Supabase setup...")
    
    try:
        from supabase import create_client
        
        # Check for service role key (preferred for uploads)
        service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        anon_key = os.getenv('SUPABASE_ANON_KEY')
        supabase_url = os.getenv('SUPABASE_URL')
        
        if service_role_key:
            logger.info("Using service role key for upload (bypasses RLS)...")
            supabase = create_client(supabase_url, service_role_key)
        elif anon_key:
            logger.info("Using anon key for upload (may require RLS compliance)...")
            supabase = create_client(supabase_url, anon_key)
        else:
            logger.error("❌ No valid Supabase key found")
            return False
        
        # Load processed data
        with open('processed_data/chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        embeddings = np.load('processed_data/embeddings.npy')
        
        logger.info(f"Uploading {len(chunks)} documents...")
        
        # Test upload first
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
                'user_id': None
            }
            
            test_result = supabase.table('documents').insert([test_doc]).execute()
            logger.info("✅ Test upload successful")
            
            # Clean up test document
            if test_result.data:
                test_id = test_result.data[0]['id']
                supabase.table('documents').delete().eq('id', test_id).execute()
                logger.info("✅ Test document cleaned up")
        
        except Exception as e:
            logger.error(f"❌ Test upload failed: {e}")
            if "401" in str(e) or "permission" in str(e).lower():
                logger.error("💡 This looks like a permissions error")
                logger.error("💡 Make sure you have SUPABASE_SERVICE_ROLE_KEY in .env")
                logger.error("💡 Or check that RLS policies allow your operation")
            return False
        
        # Upload in batches
        batch_size = 25  # Smaller batches for RLS compatibility
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        successful_uploads = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_num = i // batch_size + 1  # Define batch_num here
            
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
                        'user_id': None     # Let function handle user assignment
                    }
                    documents.append(doc)
                
                # Upload batch
                result = supabase.table('documents').insert(documents).execute()
                successful_uploads += len(documents)
                logger.info(f"✅ Uploaded batch {batch_num}/{total_batches}: {len(documents)} documents")
                
            except Exception as e:
                logger.error(f"❌ Error uploading batch {batch_num}/{total_batches}: {e}")
                
                # Show detailed error for first few failures
                if batch_num <= 3:
                    logger.error(f"Detailed error: {str(e)}")
                    if "401" in str(e) or "permission" in str(e).lower():
                        logger.error("💡 This looks like a permissions/RLS error")
                        logger.error("💡 Check your Supabase key and RLS policies")
                
                continue
        
        logger.info(f"✅ Upload completed! {successful_uploads}/{len(chunks)} documents uploaded")
        
        if successful_uploads < len(chunks):
            logger.warning(f"⚠️ {len(chunks) - successful_uploads} documents failed to upload")
            return successful_uploads > 0
        else:
            logger.info("🎉 All documents uploaded successfully!")
            return True
        
    except Exception as e:
        logger.error(f"❌ Error uploading to Supabase: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 InfinitePay AI Chatbot Quick Setup (Secure Version)")
    logger.info("Since you already have Ollama + llama3.2 installed")
    
    # Step 1: Check environment
    if not check_environment():
        logger.error("❌ Environment setup incomplete")
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies():
        logger.error("❌ Failed to install dependencies")
        return False
    
    # Step 3: Check Ollama
    if not check_ollama():
        logger.error("❌ Ollama check failed")
        return False
    
    # Step 4: Process data
    if not process_data():
        logger.error("❌ Data processing failed")
        return False
    
    # Step 5: Secure Supabase setup
    if not print_supabase_setup():
        logger.error("❌ Supabase setup incomplete")
        return False
    
    # Step 6: Upload data
    if not upload_to_supabase():
        logger.error("❌ Data upload failed")
        logger.info("💡 Try running: python upload_to_supabase.py")
        logger.info("💡 Make sure you have SUPABASE_SERVICE_ROLE_KEY in .env")
        return False
    
    logger.info("🎉 Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Start the API server: python fastapi_rag_backend.py")
    logger.info("2. Test the API: python api_test_script.py")
    logger.info("3. Visit API docs: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)