# InfinitePay Chatbot

A RAG-powered chatbot built with FastAPI, Supabase (with pgvector), and HuggingFace's models. The chatbot provides accurate answers about InfinitePay's products and services using a sophisticated document retrieval and generation system.

## 🏗️ Architecture Overview

The project consists of three main components:

1. **Data Collection (Scraper)**
2. **Document Processing & Embedding**
3. **RAG-powered Chatbot API**

### System Components

```
📁 infinitepay-chatbot/
├── 🤖 backend/          # FastAPI RAG Backend
├── 📑 data/            # Scraper & Document Processing
├── 🖥️ frontend/        # React Frontend (Optional)
└── 🎯 widget/          # Embeddable Chat Widget
```

## 🔍 Component Details

### 1. Data Collection (Scraper)

Located in `data/infinitepay_scraper.py`, the scraper:
- Crawls InfinitePay's knowledge base
- Extracts clean content from articles
- Saves content in both JSON and text formats
- Outputs:
  - `scraped_content.json`: Raw scraped data
  - `text_files/`: Individual article text files
  - `scraping_summary.json`: Metadata and statistics

Usage:
```bash
python data/infinitepay_scraper.py
```

### 2. Document Processing

Located in `data/document_processor.py`, handles:
- Text cleaning and normalization
- Content chunking with optimal overlap
- Embedding generation using sentence-transformers
- Supabase upload with vector storage

Key files:
```python
data/
├── document_processor.py      # Main processing logic
├── upload_to_supabase.py     # Database upload utilities
└── processed_data/
    ├── chunks.json           # Processed text chunks
    ├── embeddings.npy        # Generated embeddings
    └── processing_summary.json
```

### 3. Frontend Components

#### React Frontend App

Located in `frontend/`, the React application provides:
- Interactive chat interface
- Real-time response streaming
- Source citation display
- Responsive design for all devices

Key features:
```typescript
frontend/
├── src/
│   ├── ChatDemo.jsx       # Main chat interface
│   ├── assets/            # Static resources
│   └── styles/            # CSS modules
├── package.json           # Dependencies
└── vite.config.js        # Build configuration
```

#### Embeddable Chat Widget

Located in `widget/`, a lightweight embeddable version:
- Minimal dependencies
- Easy integration script
- Customizable styling
- Small bundle size

Usage:
```html
<!-- Add to any webpage -->
<script src="https://your-domain.com/widget.js"></script>
<div id="infinitepay-chat"></div>
<script>
  InfinitePayChat.init({
    apiKey: 'your-api-key',
    theme: 'light'
  });
</script>
```

### 4. RAG Backend (FastAPI)

The core chatbot service in `backend/main.py` implements:

#### RAG Pipeline
1. **Query Understanding**
   - Embedding generation using HuggingFace
   - Model: `sentence-transformers/all-MiniLM-L6-v2`

2. **Document Retrieval**
   - Vector similarity search using pgvector
   - Fallback mechanisms for robustness
   - Configurable similarity thresholds

3. **Response Generation**
   - Context-aware responses using T5
   - Model: `google/flan-t5-base`
   - Fallback templates for error cases

#### API Endpoints

```python
POST /chat
GET /health
GET /debug/embedding
GET /debug/documents
POST /admin/generate-embeddings
```

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.11+ required
python -m venv venv
source venv/bin/activate  # Unix
.\venv\Scripts\activate   # Windows
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/gsiq8/infinitepay-chatbot.git
cd infinitepay-chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Setup**
```bash
# Create .env file
cp .env.example .env

# Required variables:
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_key
HF_TOKEN=your_huggingface_token
```

4. **Run the complete setup**
```bash
python complete_setup.py
```

5. **Setup Frontend**
```bash
# Install frontend dependencies
cd frontend
npm install

# Install widget dependencies
cd ../widget
npm install
```

### Running the System

1. **Start the backend**
```bash
cd backend
uvicorn main:app --reload
```

2. **Start the frontend**
```bash
# In a new terminal
cd frontend
npm run dev

# Visit http://localhost:5173 to see the chat interface
```

3. **Build the widget** (optional)
```bash
cd widget
npm run build

# The widget will be available in dist/widget.js
```

4. **Quick test**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Como funcionam os planos?"}'
```

## 🛠️ Development

### Database Schema (Supabase)

```sql
-- documents table with pgvector extension
create table documents (
  id uuid default uuid_generate_v4() primary key,
  content text,
  page_title text,
  page_url text,
  embedding vector(384),
  metadata jsonb,
  is_public boolean default true,
  created_at timestamp with time zone default timezone('utc'::text, now())
);

-- Create the similarity search function
create or replace function match_documents (
  query_embedding vector(384),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  content text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    id,
    content,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
end;
$$;
```

### Error Handling

The system implements multiple fallback mechanisms:

1. **Vector Search Fallbacks**
   - RPC function → Manual similarity calculation → Text search
   
2. **Text Generation Fallbacks**
   - Model generation → Template-based responses → Generic message

### Monitoring

- Detailed logging with levels
- Health check endpoints
- Debug tools for embeddings
- Document status monitoring

## 📝 License

This project is licensed under the terms of the LICENSE file included in the repository.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 🔧 Troubleshooting

Common issues and solutions:

1. **Embedding Generation Fails**
   - Check HF_TOKEN validity
   - Verify model accessibility
   - Check input text length

2. **Vector Search Issues**
   - Verify Supabase connection
   - Check pgvector extension
   - Validate embedding dimensions

3. **Text Generation Timeout**
   - Adjust timeout settings
   - Check model availability
   - Consider fallback models

## 📚 API Documentation

Full API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
