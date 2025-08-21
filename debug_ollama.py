#!/usr/bin/env python3
"""
Debug script to check Ollama setup and fix model issues
"""

import httpx
import json
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def check_ollama_status():
    """Check Ollama server status and available models"""
    print("🔍 Checking Ollama status...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check if Ollama is running
            response = await client.get("http://localhost:11434/api/tags")
            
            if response.status_code != 200:
                print(f"❌ Ollama not responding: {response.status_code}")
                return False
            
            data = response.json()
            models = data.get('models', [])
            
            print(f"✅ Ollama is running with {len(models)} models:")
            for model in models:
                name = model.get('name', 'Unknown')
                size = model.get('size', 0)
                print(f"   - {name} ({size / 1024 / 1024 / 1024:.1f}GB)")
            
            return models
            
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

async def test_model_generation(model_name):
    """Test model generation with a simple prompt"""
    print(f"\n🧪 Testing model generation with: {model_name}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hello, how are you?",
                    "stream": False
                }
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', 'No response')
                print(f"✅ Generation successful!")
                print(f"Response: {generated_text[:100]}...")
                return True
            else:
                print(f"❌ Generation failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing generation: {e}")
        return False

async def test_chat_endpoint():
    """Test the chat endpoint to see the exact error"""
    print("\n🔍 Testing chat endpoint directly...")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8000/chat",
                headers={"Authorization": "Bearer test-key"},
                json={"message": "Hello test"}
            )
            
            print(f"Chat endpoint status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Chat endpoint working!")
                print(f"Response: {data.get('response', 'No response')[:100]}...")
            else:
                print(f"❌ Chat endpoint failed: {response.status_code}")
                print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing chat endpoint: {e}")

async def main():
    """Main debug function"""
    print("🔧 Ollama Debug Script")
    print("=" * 50)
    
    # Check Ollama status
    models = await check_ollama_status()
    if not models:
        print("\n💡 Solutions:")
        print("1. Start Ollama: 'ollama serve'")
        print("2. Install a model: 'ollama pull llama3.2'")
        return
    
    # Find best model to use
    model_candidates = []
    for model in models:
        name = model.get('name', '')
        if any(keyword in name.lower() for keyword in ['llama3.2', 'llama3', 'llama2']):
            model_candidates.append(name)
    
    if not model_candidates:
        print("\n❌ No suitable models found!")
        print("Available models don't include llama3.2, llama3, or llama2")
        print("\n💡 Install a suitable model:")
        print("ollama pull llama3.2")
        return
    
    # Test the first suitable model
    best_model = model_candidates[0]
    print(f"\n🎯 Best model found: {best_model}")
    
    # Test generation
    success = await test_model_generation(best_model)
    
    if success:
        print(f"\n✅ Model {best_model} is working!")
        
        # Update environment variable
        current_model = os.getenv('OLLAMA_MODEL', 'llama3.2:latest')
        if current_model != best_model:
            print(f"\n💡 Consider updating your .env file:")
            print(f"Current OLLAMA_MODEL: {current_model}")
            print(f"Recommended OLLAMA_MODEL: {best_model}")
            
            # Try to update .env file
            try:
                with open('.env', 'r') as f:
                    content = f.read()
                
                if 'OLLAMA_MODEL=' in content:
                    # Update existing line
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith('OLLAMA_MODEL='):
                            lines[i] = f'OLLAMA_MODEL={best_model}'
                            break
                    content = '\n'.join(lines)
                else:
                    # Add new line
                    content += f'\nOLLAMA_MODEL={best_model}\n'
                
                with open('.env', 'w') as f:
                    f.write(content)
                
                print(f"✅ Updated .env file with OLLAMA_MODEL={best_model}")
                
            except Exception as e:
                print(f"⚠️ Could not update .env file: {e}")
                print(f"Please manually add: OLLAMA_MODEL={best_model}")
    
    # Test chat endpoint if it's running
    print("\n" + "=" * 50)
    await test_chat_endpoint()
    
    print("\n🎯 Summary:")
    if success:
        print(f"✅ Ollama is working with model: {best_model}")
        print("✅ You can restart your FastAPI server now")
    else:
        print("❌ Ollama model issues detected")
        print("💡 Try installing a different model or check Ollama logs")

if __name__ == "__main__":
    asyncio.run(main())