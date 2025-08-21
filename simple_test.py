#!/usr/bin/env python3
"""
Simple test to verify chatbot is working after fixes
"""

import asyncio
import httpx
import json

async def test_chatbot():
    """Test the chatbot with a simple question"""
    print("🧪 Testing InfinitePay AI Chatbot...")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test health first
            health_response = await client.get("http://localhost:8000/health")
            
            if health_response.status_code != 200:
                print(f"❌ Health check failed: {health_response.status_code}")
                print("💡 Make sure the server is running: python fixed_run_chatbot.py")
                return False
            
            health_data = health_response.json()
            print(f"✅ Server health: {health_data['status']}")
            print(f"   Services: {health_data['services']}")
            
            # Test chat
            print("\n🗣️ Testing chat with: 'Como funciona o Pix?'")
            
            chat_response = await client.post(
                "http://localhost:8000/chat",
                headers={"Authorization": "Bearer test-key"},
                json={"message": "Como funciona o Pix?"}
            )
            
            if chat_response.status_code == 200:
                chat_data = chat_response.json()
                response_text = chat_data.get('response', '')
                sources = chat_data.get('sources', [])
                
                print(f"✅ Chat response received!")
                print(f"📝 Response: {response_text}")
                print(f"📚 Sources: {len(sources)} documents")
                
                if sources:
                    print("   Top source:")
                    top_source = sources[0]
                    print(f"   - Title: {top_source.get('title', 'N/A')}")
                    print(f"   - Similarity: {top_source.get('similarity', 0):.3f}")
                
                # Check if we got a real response (not an error message)
                if "erro" in response_text.lower() or "desculpe" in response_text.lower():
                    print("⚠️ Got error response - there might still be issues")
                    return False
                else:
                    print("🎉 Everything looks good!")
                    return True
            else:
                print(f"❌ Chat request failed: {chat_response.status_code}")
                print(f"Response: {chat_response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def main():
    print("🔍 Simple Chatbot Test")
    print("=" * 40)
    
    success = await test_chatbot()
    
    if success:
        print("\n✅ Test passed! Your chatbot is working correctly.")
    else:
        print("\n❌ Test failed. Please check the issues above.")
        print("\n💡 Troubleshooting steps:")
        print("1. Make sure server is running: python fixed_run_chatbot.py")
        print("2. Run the fixer: python fix_data_and_model.py")
        print("3. Check Ollama: ollama list")

if __name__ == "__main__":
    asyncio.run(main())