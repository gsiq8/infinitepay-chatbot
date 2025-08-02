# install_dependencies.sh
#!/bin/bash

echo "🚀 Setting up InfinitePay Chatbot Development Environment"
echo "=================================================="

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Download sentence transformer model (pre-download for faster startup)
echo "🤖 Pre-downloading embedding model..."
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Check if Ollama is installed and running
echo "🦙 Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama found in system"
    
    # Check if Ollama service is running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "✅ Ollama service is running"
        
        # List available models
        echo "📋 Available Ollama models:"
        ollama list
        
        # Check for recommended models
        if ollama list | grep -q "llama3.2"; then
            echo "✅ Llama 3.2 found - ready to use!"
        elif ollama list | grep -q "llama3.1"; then
            echo "✅ Llama 3.1 found - will use as fallback"
        else
            echo "⚠️  No Llama 3.x models found. You can install one with:"
            echo "   ollama pull llama3.2:3b"
            echo "   or"
            echo "   ollama pull llama3.1:8b"
        fi
    else
        echo "⚠️  Ollama installed but not running. Start with:"
        echo "   ollama serve"
    fi
else
    echo "⚠️  Ollama not found. Install from: https://ollama.ai"
    echo "   After installation, run: ollama pull llama3.2:3b"
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p infinitepay_data
mkdir -p processed_data
mkdir -p logs
mkdir -p tests

# Create environment file template
echo "🔧 Creating .env template..."
cat > .env.example << EOL
# Database Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.1:8b

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=60
MAX_REQUESTS_PER_HOUR=1000

# Logging
LOG_LEVEL=INFO
EOL

echo "✅ Environment template created. Copy .env.example to .env and fill in your values."

# Create basic project structure
echo "📝 Creating project structure..."
cat > run_scraper.py << 'EOL'
#!/usr/bin/env python3
"""
Quick script to run the InfinitePay scraper
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infinitepay_scraper import main

if __name__ == "__main__":
    main()
EOL

cat > run_processor.py << 'EOL'
#!/usr/bin/env python3
"""
Quick script to run the document processor
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processor import main

if __name__ == "__main__":
    main()
EOL

# Make scripts executable
chmod +x run_scraper.py
chmod +x run_processor.py

echo "🎯 Setup completed!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Copy .env.example to .env and configure"
echo "3. Run scraper: python run_scraper.py"
echo "4. Process documents: python run_processor.py"
echo ""
echo "📖 For detailed instructions, see docs/implementation-guide.md"

---

# Makefile
.PHONY: install scrape process test clean help

# Default target
help:
	@echo "InfinitePay Chatbot - Available Commands:"
	@echo "========================================"
	@echo "install     - Install dependencies and setup environment"
	@echo "scrape      - Scrape InfinitePay website content"
	@echo "process     - Process scraped content for RAG"
	@echo "test        - Run tests"
	@echo "clean       - Clean generated files"
	@echo "dev         - Start development environment"
	@echo "demo        - Run demo server"

install:
	@echo "🚀 Installing dependencies..."
	@chmod +x install_dependencies.sh
	@./install_dependencies.sh

scrape:
	@echo "🕷️ Scraping InfinitePay website..."
	@python run_scraper.py

process:
	@echo "⚙️ Processing documents..."
	@python run_processor.py

pipeline: scrape process
	@echo "✅ Full data pipeline completed!"

test:
	@echo "🧪 Running tests..."
	@python -m pytest tests/ -v

clean:
	@echo "🧹 Cleaning generated files..."
	@rm -rf infinitepay_data/
	@rm -rf processed_data/
	@rm -rf __pycache__/
	@rm -rf *.pyc
	@rm -rf .pytest_cache/
	@echo "✅ Cleanup completed!"

dev:
	@echo "🔧 Starting development environment..."
	@source venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000

demo:
	@echo "🎬 Starting demo server..."
	@source venv/bin/activate && python demo_server.py

---

# .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data files
infinitepay_data/
processed_data/
logs/
*.log

# Environment variables
.env
.env.local
.env.production

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Jupyter Notebook
.ipynb_checkpoints

# Model files (too large for git)
*.model
*.bin
models/

# Test coverage
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

---

# docker-compose.yml
version: '3.8'

services:
  # Ollama for local LLM
  ollama:
    image: ollama/ollama:latest
    container_name: infinitepay_ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_ORIGINS=*
    restart: unless-stopped

  # PostgreSQL with pgvector (alternative to Supabase for local dev)
  postgres:
    image: pgvector/pgvector:pg15
    container_name: infinitepay_postgres
    environment:
      POSTGRES_DB: infinitepay_chatbot
      POSTGRES_USER: chatbot_user
      POSTGRES_PASSWORD: chatbot_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  # Redis for caching (optional)
  redis:
    image: redis:7-alpine
    container_name: infinitepay_redis
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  ollama_data:
  postgres_data: