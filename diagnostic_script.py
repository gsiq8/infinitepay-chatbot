#!/usr/bin/env python3
"""
Diagnostic script to identify FastAPI chatbot issues
Run this to check all dependencies and configurations
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Python Version Check")
    print(f"   Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    
    if major == 3 and minor >= 13:
        print("   ⚠️  WARNING: Python 3.13+ may have compatibility issues")
        print("   💡 Consider using Python 3.11 or 3.12 for better compatibility")
    else:
        print("   ✅ Python version looks good")
    print()

def check_imports():
    """Check if all required packages can be imported"""
    print("📦 Package Import Check")
    
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('supabase', 'Supabase client'),
        ('sentence_transformers', 'SentenceTransformer'),
        ('httpx', 'HTTPX'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('huggingface_hub', 'Hugging Face Hub')
    ]
    
    failed_imports = []
    
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {description}")
        except ImportError as e:
            print(f"   ❌ {description}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n   🔧 Failed imports: {', '.join(failed_imports)}")
        print("   💡 Run the dependency fix script first")
    else:
        print("   🎉 All packages imported successfully!")
    print()
    
    return len(failed_imports) == 0

def check_ollama():
    """Check if Ollama is running and accessible"""
    print("🦙 Ollama Check")
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"   ✅ Ollama is running with {len(models)} models")
            for model in models:
                print(f"      - {model.get('name', 'Unknown')}")
        else:
            print(f"   ❌ Ollama responded with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Ollama connection failed: {e}")
        print("   💡 Start Ollama with: ollama serve")
    print()

def check_environment():
    """Check environment variables and configuration"""
    print("🔧 Environment Check")
    
    env_file = Path(".env")
    if env_file.exists():
        print("   ✅ .env file found")
        with open(env_file) as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    key = line.split('=')[0]
                    print(f"      - {key}")
    else:
        print("   ⚠️  .env file not found")
        print("   💡 Create .env with your Supabase credentials")
    
    # Check if required env vars are set
    required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY']
    for var in required_vars:
        if os.getenv(var):
            print(f"   ✅ {var} is set")
        else:
            print(f"   ❌ {var} is not set")
    print()

def check_files():
    """Check if required files exist"""
    print("📁 File Check")
    
    required_files = [
        'fastapi_rag_backend.py',
        'api_test_script.py',
        'run_chatbot.py'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} not found")
    print()

def check_port_availability():
    """Check if port 8000 is available"""
    print("🌐 Port Check")
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', 8000))
            if result == 0:
                print("   ⚠️  Port 8000 is already in use")
                print("   💡 Stop the existing server or use a different port")
            else:
                print("   ✅ Port 8000 is available")
    except Exception as e:
        print(f"   ❌ Port check failed: {e}")
    print()

def suggest_fixes():
    """Provide fix suggestions"""
    print("🔧 Suggested Fixes")
    print("   1. Install/fix dependencies:")
    print("      pip install --upgrade pip setuptools wheel")
    print("      pip install 'numpy>=1.26.0'")
    print("      pip install 'huggingface_hub>=0.19.0,<0.21.0'")
    print("      pip install 'sentence-transformers>=2.2.2,<3.0.0'")
    print()
    print("   2. If using Python 3.13, consider downgrading:")
    print("      python3.11 -m venv venv_py311")
    print("      source venv_py311/bin/activate")
    print()
    print("   3. Start Ollama if not running:")
    print("      ollama serve")
    print()
    print("   4. Check your .env file has correct Supabase credentials")
    print()

def main():
    print("🔍 InfinitePay Chatbot Diagnostic Tool")
    print("=" * 50)
    
    check_python_version()
    imports_ok = check_imports()
    check_ollama()
    check_environment()
    check_files()
    check_port_availability()
    
    if not imports_ok:
        suggest_fixes()
    else:
        print("🎉 Basic checks passed! Try running the FastAPI server now.")
        print("   python fastapi_rag_backend.py")

if __name__ == "__main__":
    main()