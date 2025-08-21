"""
Upload processed data to Supabase
This script takes the processed chunks and embeddings and uploads them to Supabase.
"""

from dotenv import load_dotenv
import os
import json
import numpy as np
from supabase import create_client, Client
import logging
from tqdm import tqdm
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(processed_data_dir: str):
    """Load chunks and embeddings from the processed data directory."""
    try:
        # Load chunks
        with open(os.path.join(processed_data_dir, 'chunks.json'), 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"✅ Loaded {len(chunks)} chunks from chunks.json")

        # Load embeddings
        embeddings = np.load(os.path.join(processed_data_dir, 'embeddings.npy'))
        logger.info(f"✅ Loaded embeddings with shape {embeddings.shape}")

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")

        return chunks, embeddings

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

def init_supabase() -> Client:
    """Initialize Supabase client."""
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

        if not supabase_url or not supabase_key:
            raise ValueError("Missing Supabase credentials")

        client = create_client(supabase_url, supabase_key)
        logger.info("✅ Supabase client initialized")
        return client

    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")
        sys.exit(1)

def clear_existing_data(supabase: Client):
    """Clear existing documents from the database."""
    try:
        result = supabase.table('documents').delete().execute()
        logger.info("✅ Cleared existing documents from database")
    except Exception as e:
        logger.error(f"Failed to clear existing data: {e}")
        sys.exit(1)

def upload_documents(supabase: Client, chunks: list, embeddings: np.ndarray):
    """Upload documents with their embeddings to Supabase."""
    try:
        for i, (chunk, embedding) in enumerate(tqdm(zip(chunks, embeddings), total=len(chunks))):
            # Prepare document data
            doc_data = {
                'page_title': chunk['title'],
                'content': chunk['content'],
                'page_url': chunk['url'],
                'embedding': embedding.tolist()  # Convert numpy array to list
            }

            # Upload to Supabase
            result = supabase.table('documents').insert(doc_data).execute()

        logger.info(f"✅ Successfully uploaded {len(chunks)} documents to Supabase")

    except Exception as e:
        logger.error(f"Failed to upload documents: {e}")
        sys.exit(1)

def main():
    """Main execution function."""
    # Load environment variables
    load_dotenv()

    # Set up paths
    current_dir = Path(__file__).parent
    processed_data_dir = current_dir / 'processed_data'

    # Initialize Supabase
    supabase = init_supabase()

    # Load data
    chunks, embeddings = load_data(str(processed_data_dir))

    # Clear existing data (optional, uncomment if needed)
    # clear_existing_data(supabase)

    # Upload documents
    upload_documents(supabase, chunks, embeddings)

if __name__ == "__main__":
    main()
