"""
Complete Setup Script for InfinitePay AI Chatbot
Handles data processing, Supabase setup, and testing
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'supabase',
        'sentence_transformers',
        'fastapi',
        'uvicorn',
        'httpx',
        'numpy',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_ANON_KEY'
        'HF_TOKEN'
    ]
    
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        logger.error("Please set these variables in your .env file")
        return False
    
    return True

def process_scraped_data():
    """Process scraped data and create embeddings if not already done"""
    from sentence_transformers import SentenceTransformer
    
    # Check if processed data already exists
    if os.path.exists("processed_data/chunks.json") and os.path.exists("processed_data/embeddings.npy"):
        logger.info("Processed data already exists, skipping processing")
        return True
    
    # Check if scraped data exists
    if not os.path.exists("infinitepay_data/text_files"):
        logger.error("No scraped data found. Please run the scraper first.")
        return False
    
    logger.info("Processing scraped data...")
    
    # Initialize sentence transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Process text files
    chunks = []
    texts_for_embedding = []
    
    text_files_dir = "infinitepay_data/text_files"
    
    for filename in os.listdir(text_files_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(text_files_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:
                    # Extract page info from filename
                    page_title = filename.replace('.txt', '').replace('page_', '').replace('_', ' ')
                    
                    # Split content into chunks (simple approach)
                    chunk_size = 500
                    words = content.split()
                    
                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i:i + chunk_size]
                        chunk_text = ' '.join(chunk_words)
                        
                        if len(chunk_text.strip()) > 50:  # Only process meaningful chunks
                            chunk = {
                                'content': chunk_text,
                                'metadata': {
                                    'source_file': filename,
                                    'chunk_index': i // chunk_size
                                },
                                'page_title': page_title,
                                'page_url': f"https://infinitepay.io/{filename.replace('.txt', '').replace('page_', '')}",
                                'chunk_index': i // chunk_size
                            }
                            
                            chunks.append(chunk)
                            texts_for_embedding.append(chunk_text)
            
            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")
    
    if not chunks:
        logger.error("No valid chunks created from scraped data")
        return False
    
    logger.info(f"Created {len(chunks)} chunks, generating embeddings...")
    
    # Generate embeddings
    embeddings = model.encode(texts_for_embedding, show_progress_bar=True)
    
    # Save processed data
    os.makedirs("processed_data", exist_ok=True)
    
    with open("processed_data/chunks.json", 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    np.save("processed_data/embeddings.npy", embeddings)
    
    # Save processing summary
    summary = {
        'total_chunks': len(chunks),
        'embedding_dimensions': embeddings.shape[1],
        'model_used': 'all-MiniLM-L6-v2',
        'chunk_size': chunk_size
    }
    
    with open("processed_data/processing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Processing completed! Created {len(chunks)} chunks with embeddings")
    return True

def setup_supabase():
    """Setup Supabase tables and upload data"""
    from supabase import create_client
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    
    supabase = create_client(supabase_url, supabase_key)
    
    # Print SQL commands for manual execution
    logger.info("Please run the following SQL commands in your Supabase SQL editor:")
    logger.info("="*60)
    
    sql_commands = """
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(384), -- all-MiniLM-L6-v2 produces 384-dim vectors
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    page_title TEXT,
    page_url TEXT,
    chunk_index INTEGER
);

-- Create an index for vector similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for metadata queries
CREATE INDEX IF NOT EXISTS documents_metadata_idx 
ON documents USING GIN (metadata);

-- Create similarity search function
CREATE OR REPLACE FUNCTION similarity_search(
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
LANGUAGE sql STABLE
AS $
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        documents.page_title,
        documents.page_url,
        1 - (documents.embedding <=> query_embedding) as similarity
    FROM documents
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
$;
"""
    
    print(sql_commands)
    logger.info("="*60)
    
    # Wait for user confirmation
    confirmation = input("Have you run the SQL commands in Supabase? (y/n): ")
    if confirmation.lower() != 'y':
        logger.info("Please run the SQL commands first, then restart this script")
        return False
    
    # Load processed data
    try:
        with open("processed_data/chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        embeddings = np.load("processed_data/embeddings.npy")
        
        logger.info(f"Loaded {len(chunks)} chunks for upload")
        
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        return False
    
    # Upload data in batches
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        
        documents_to_insert = []
        
        for chunk, embedding in zip(batch_chunks, batch_embeddings):
            doc = {
                'content': chunk['content'],
                'metadata': chunk.get('metadata', {}),
                'embedding': embedding.tolist(),
                'page_title': chunk.get('page_title', ''),
                'page_url': chunk.get('page_url', ''),
                'chunk_index': chunk.get('chunk_index', 0)
            }
            documents_to_insert.append(doc)
        
        try:
            result = supabase.table('documents').insert(documents_to_insert).execute()
            batch_num = i // batch_size + 1
            logger.info(f"Uploaded batch {batch_num}/{total_batches}: {len(documents_to_insert)} documents")
            
        except Exception as e:
            logger.error(f"Error uploading batch {batch_num}: {e}")
            return False
    
    logger.info("Data upload completed successfully!")
    return True

def hf_setup():
    """Setup Hugging Face client"""
    from huggingface_hub import InferenceClient
    
    hf_token = os.getenv('HF_TOKEN')
    
    if hf_token:
        try:
            hf_client = InferenceClient(
                provider="auto",
                api_key=hf_token
            )
            logger.info("✅ Hugging Face Inference Client initialized")
        except Exception as e:
            logger.error(f"❌ Hugging Face Inference Client initialization failed: {e}")
    else:
        logger.warning("⚠️ HF_TOKEN not found. AI features will be disabled.")

    logger.info("🎉 Services initialization complete.")

def test_system():
    """Test the complete system"""
    import asyncio
    import httpx
    
    logger.info("Testing the complete system...")
    
    # Test Supabase connection
    try:
        from supabase import create_client
        
        supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))
        result = supabase.table('documents').select('id').limit(1).execute()
        
        if result.data:
            logger.info("✅ Supabase connection successful")
        else:
            logger.warning("⚠️ Supabase connected but no data found")
            
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        return False
    
    # Test vector search
    try:
        logger.info("Testing vector search...")
        from huggingface_hub import InferenceClient
        
        model = 'SentenceTransformer/all-MiniLM-L6-v2'
        query_embedding = model.encode("Como funciona o pagamento").tolist()

        result = supabase.rpc(
            'similarity_search',
            {
                'query_embedding': query_embedding,
                'match_count': 3
            }
        ).execute()
        
        if result.data:
            logger.info("✅ Vector search working")
            logger.info(f"Found {len(result.data)} relevant documents")
        else:
            logger.warning("⚠️ Vector search returned no results")
            
    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")
        return False
    
    logger.info("🎉 Similarity test completed successfully!")
    return True

def create_sample_queries():
    """Create sample queries for testing"""
    sample_queries = [
        "Como funciona o pagamento com link?",
        "Quais são as vantagens do Pix com cartão de crédito?",
        "Como funcionam os planos?",
        "Qual a menor taxa de maquininha?",
    ]
    
    os.makedirs("processed_data", exist_ok=True)
    
    with open("processed_data/sample_queries.json", 'w', encoding='utf-8') as f:
        json.dump(sample_queries, f, ensure_ascii=False, indent=2)
    
    logger.info("Sample queries created for testing")

def main():
    """Main setup function"""
    logger.info("🚀 Starting InfinitePay AI Chatbot Setup")
    
    # Step 1: Check dependencies
    logger.info("Step 1: Checking dependencies...")
    if not check_dependencies():
        logger.error("Dependencies check failed")
        sys.exit(1)
    
    # Step 2: Check environment
    logger.info("Step 2: Checking environment variables...")
    if not check_environment():
        logger.error("Environment check failed")
        sys.exit(1)
    
    
    # Step 3: Process data
    logger.info("Step 4: Processing scraped data...")
    if not process_scraped_data():
        logger.error("Data processing failed")
        sys.exit(1)
    
    # Step 4: Setup Supabase
    logger.info("Step 5: Setting up Supabase...")
    if not setup_supabase():
        logger.error("Supabase setup failed")
        sys.exit(1)
    
    # Step 5: Create sample queries
    logger.info("Step 6: Creating sample queries...")
    create_sample_queries()
    
    # Step 6: Test Hugging Face client
    if not hf_setup():
        logger.error("Hugging Face client setup failed")
        sys.exit(1)
    
    # Step 7: Similarity search test
    logger.info("Step 7: Testing similarity search...")
    # Step 5: Test system
    logger.info("Step 7: Testing complete system...")
    if not test_system():
        logger.error("System test failed")
        sys.exit(1)
    
    logger.info("🎉 Setup completed successfully!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Start the FastAPI backend: python fastapi_rag_backend.py")
    logger.info("2. Test the API at: http://localhost:8000/docs")
    logger.info("3. Try sample queries with: http://localhost:8000/chat")
    logger.info("")
    logger.info("Your InfinitePay AI Chatbot is ready! 🤖")

if __name__ == "__main__":
    main()