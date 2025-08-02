import json
import os
from pathlib import Path
from typing import List, Dict, Any
import re
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    id: str
    content: str
    title: str
    url: str
    language: str
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: np.ndarray = None

class DocumentProcessor:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize document processor with embedding model
        all-MiniLM-L6-v2 is lightweight and works well for Portuguese/English
        """
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks: List[DocumentChunk] = []
        
        # Fintech-specific sensitive patterns (more comprehensive)
        self.sensitive_patterns = [
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit cards
            r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',  # CPF-like patterns
            r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b',  # CNPJ patterns
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'\bR\$\s*\d{1,3}(?:\.\d{3})*(?:,\d{2})?\b',  # Brazilian currency amounts
            r'\b(?:senha|password|token|key|secret|api[_-]?key)\s*[:=]\s*\S+',  # Credentials
            r'\b(?:bearer|basic)\s+[a-zA-Z0-9+/=]+\b',  # Auth tokens
            r'\bmastercard|visa|amex|american\s+express\b',  # Card brands in sensitive context
        ]
        
        # Quality filters
        self.min_chunk_length = 50
        self.max_chunk_length = 1000
        self.overlap_size = 100
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\u00C0-\u017F\u0100-\u024F]', '', text)
        
        # Remove very long words (likely garbage)
        words = text.split()
        words = [word for word in words if len(word) <= 50]
        text = ' '.join(words)
        
        return text.strip()
    
    def apply_security_filters(self, text: str) -> str:
        """Apply security filters to remove sensitive information"""
        filtered_text = text
        
        for pattern in self.sensitive_patterns:
            filtered_text = re.sub(pattern, '[DADOS_PROTEGIDOS]', filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    def split_into_chunks(self, text: str, title: str, url: str, metadata: Dict) -> List[str]:
        """Split text into overlapping chunks for better context retrieval"""
        # Clean text first
        text = self.clean_text(text)
        text = self.apply_security_filters(text)
        
        if len(text) < self.min_chunk_length:
            return []
        
        chunks = []
        words = text.split()
        
        # Calculate words per chunk (approximate)
        words_per_chunk = self.max_chunk_length // 6  # Rough estimate: 6 chars per word
        overlap_words = self.overlap_size // 6
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text) >= self.min_chunk_length:
                chunks.append(chunk_text)
            
            # Stop if we've reached the end
            if i + words_per_chunk >= len(words):
                break
        
        return chunks
    
    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract key terms from text using TF-IDF"""
        try:
            # Simple keyword extraction for Portuguese/English
            vectorizer = TfidfVectorizer(
                max_features=num_keywords,
                stop_words=None,  # We'll handle stop words manually
                ngram_range=(1, 2),
                min_df=1
            )
            
            # Portuguese and English stop words (simplified)
            stop_words = {
                'de', 'da', 'do', 'das', 'dos', 'e', 'em', 'na', 'no', 'nas', 'nos',
                'para', 'por', 'com', 'um', 'uma', 'os', 'as', 'que', 'se',
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has'
            }
            
            # Filter out stop words
            words = text.lower().split()
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            filtered_text = ' '.join(filtered_words)
            
            if not filtered_text.strip():
                return []
            
            tfidf_matrix = vectorizer.fit_transform([filtered_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [kw[0] for kw in keyword_scores[:num_keywords]]
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    def process_scraped_data(self, input_file: str) -> List[DocumentChunk]:
        """Process scraped JSON data into document chunks"""
        logger.info(f"Processing scraped data from: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        
        chunk_id = 0
        
        for doc in scraped_data:
            title = doc.get('title', 'Untitled')
            url = doc.get('url', '')
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            language = metadata.get('lang', 'pt')
            
            # Skip if content is too short
            if len(content.strip()) < self.min_chunk_length:
                continue
            
            # Split into chunks
            chunks = self.split_into_chunks(content, title, url, metadata)
            
            for chunk_index, chunk_content in enumerate(chunks):
                # Extract keywords for this chunk
                keywords = self.extract_keywords(chunk_content)
                
                # Create enhanced metadata
                enhanced_metadata = {
                    **metadata,
                    'keywords': keywords,
                    'word_count': len(chunk_content.split()),
                    'char_count': len(chunk_content),
                    'has_financial_terms': any(term in chunk_content.lower() for term in 
                                              ['pagamento', 'payment', 'cartão', 'card', 'pix', 'banco', 'bank',
                                               'taxa', 'fee', 'transação', 'transaction', 'dinheiro', 'money']),
                    'chunk_type': 'content'
                }
                
                document_chunk = DocumentChunk(
                    id=f"chunk_{chunk_id:04d}",
                    content=chunk_content,
                    title=title,
                    url=url,
                    language=language,
                    chunk_index=chunk_index,
                    metadata=enhanced_metadata
                )
                
                self.chunks.append(document_chunk)
                chunk_id += 1
        
        logger.info(f"Created {len(self.chunks)} document chunks")
        return self.chunks
    
    def generate_embeddings(self) -> None:
        """Generate embeddings for all chunks"""
        logger.info("Generating embeddings for document chunks...")
        
        if not self.chunks:
            logger.error("No chunks to process. Run process_scraped_data first.")
            return
        
        # Prepare texts for embedding
        texts = []
        for chunk in self.chunks:
            # Combine title and content for better context
            combined_text = f"{chunk.title}\n\n{chunk.content}"
            texts.append(combined_text)
        
        # Generate embeddings in batches for efficiency
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=True,
                batch_size=batch_size
            )
            all_embeddings.extend(batch_embeddings)
        
        # Assign embeddings to chunks
        for chunk, embedding in zip(self.chunks, all_embeddings):
            chunk.embedding = embedding
        
        logger.info(f"Generated embeddings for {len(self.chunks)} chunks")
    
    def save_processed_data(self, output_dir: str = "processed_data") -> None:
        """Save processed chunks and embeddings"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save chunks with embeddings
        chunks_data = []
        embeddings = []
        
        for chunk in self.chunks:
            chunk_dict = {
                'id': chunk.id,
                'content': chunk.content,
                'title': chunk.title,
                'url': chunk.url,
                'language': chunk.language,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata
            }
            chunks_data.append(chunk_dict)
            embeddings.append(chunk.embedding)
        
        # Save chunks metadata
        with open(f"{output_dir}/chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        # Save embeddings separately (binary format for efficiency)
        embeddings_array = np.array(embeddings)
        np.save(f"{output_dir}/embeddings.npy", embeddings_array)
        
        # Save processing summary
        summary = {
            'total_chunks': len(self.chunks),
            'embedding_dimension': embeddings_array.shape[1] if embeddings_array.size > 0 else 0,
            'languages': list(set(chunk.language for chunk in self.chunks)),
            'avg_chunk_length': np.mean([len(chunk.content.split()) for chunk in self.chunks]),
            'total_documents': len(set(chunk.url for chunk in self.chunks)),
            'has_financial_content': sum(1 for chunk in self.chunks 
                                       if chunk.metadata.get('has_financial_terms', False)),
            'embedding_model': self.embedding_model.get_sentence_embedding_dimension()
        }
        
        with open(f"{output_dir}/processing_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed data saved to {output_dir}/")
        logger.info(f"Summary: {summary}")
    
    def create_sample_queries(self, output_dir: str = "processed_data") -> None:
        """Create sample queries for testing the RAG system"""
        sample_queries = {
            "portuguese": [
                "Como funciona o PIX no InfinitePay?",
                "Quais são as taxas de transação?",
                "Como posso integrar o sistema de pagamentos?",
                "O que fazer se minha transação foi negada?",
                "Como funciona a segurança dos pagamentos?",
                "Quais cartões são aceitos?",
                "Como receber pagamentos online?",
                "Qual o limite de transações por dia?",
                "Como consultar minhas vendas?",
                "O InfinitePay tem suporte 24 horas?"
            ],
            "english": [
                "How does PIX work with InfinitePay?",
                "What are the transaction fees?",
                "How can I integrate the payment system?",
                "What should I do if my transaction was declined?",
                "How does payment security work?",
                "Which cards are accepted?",
                "How to receive online payments?",
                "What's the daily transaction limit?",
                "How to check my sales?",
                "Does InfinitePay have 24/7 support?"
            ]
        }
        
        with open(f"{output_dir}/sample_queries.json", 'w', encoding='utf-8') as f:
            json.dump(sample_queries, f, ensure_ascii=False, indent=2)
        
        logger.info("Sample queries created for testing")

def main():
    """Main processing function"""
    # Initialize processor
    processor = DocumentProcessor()
    
    # Process scraped data
    input_file = "infinitepay_data/scraped_content.json"
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please run the scraper first: python infinitepay_scraper.py")
        return
    
    # Process documents
    chunks = processor.process_scraped_data(input_file)
    
    if not chunks:
        logger.error("No chunks were created. Check your input data.")
        return
    
    # Generate embeddings
    processor.generate_embeddings()
    
    # Save processed data
    processor.save_processed_data("processed_data")
    
    # Create sample queries for testing
    processor.create_sample_queries("processed_data")
    
    print(f"\n✅ Document processing completed!")
    print(f"📄 {len(chunks)} chunks created")
    print(f"🔍 Embeddings generated")
    print(f"📁 Data saved to 'processed_data/' directory")
    print(f"🧪 Sample queries created for testing")

if __name__ == "__main__":
    main()