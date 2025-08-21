#!/usr/bin/env python3
"""
Fixed run script for InfinitePay AI Chatbot
Addresses server startup detection issues
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path
import logging
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatbotRunner:
    def __init__(self):
        self.server_process = None
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("\n🛑 Shutting down chatbot...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up server process"""
        if self.server_process and self.server_process.poll() is None:
            logger.info("Stopping FastAPI server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
    
    def check_setup(self):
        """Verify setup is complete"""
        logger.info("Checking setup...")
        
        # Check required files
        required_files = [
            '.env',
            'fastapi_rag_backend.py'
        ]
        
        # Check optional files (warn if missing)
        optional_files = [
            'processed_data/chunks.json',
            'processed_data/embeddings.npy'
        ]
        
        missing_required = [f for f in required_files if not Path(f).exists()]
        missing_optional = [f for f in optional_files if not Path(f).exists()]
        
        if missing_required:
            logger.error(f"❌ Missing required files: {missing_required}")
            return False
        
        if missing_optional:
            logger.warning(f"⚠️ Missing optional files: {missing_optional}")
            logger.warning("The API will work but with limited functionality")
        
        # Check environment variables
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
        
        if not supabase_url or not supabase_key:
            logger.error("❌ Missing Supabase credentials in .env")
            logger.error("Need SUPABASE_URL and either SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY")
            return False
        
        logger.info("✅ Setup verified")
        return True
    
    def check_services(self):
        """Check external services"""
        logger.info("Checking external services...")
        
        # Check Ollama
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    
                    if models:
                        logger.info(f"✅ Ollama ready with {len(models)} models")
                        logger.info(f"   Available: {', '.join(models[:3])}...")
                    else:
                        logger.error("❌ No Ollama models found")
                        logger.error("Please install a model: ollama pull llama3.2")
                        return False
                else:
                    logger.error("❌ Ollama not responding")
                    return False
        except Exception as e:
            logger.error(f"❌ Ollama check failed: {e}")
            logger.error("Please ensure Ollama is running: ollama serve")
            return False
        
        # Check Supabase connection
        try:
            from supabase import create_client
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
            
            supabase = create_client(supabase_url, supabase_key)
            result = supabase.table('documents').select('id').limit(1).execute()
            logger.info("✅ Supabase connection verified")
            
            if not result.data:
                logger.warning("⚠️ No data found in Supabase documents table")
                logger.warning("You may need to run the upload step again")
            
        except Exception as e:
            logger.error(f"❌ Supabase check failed: {e}")
            return False
        
        return True
    
    def start_server(self):
        """Start the FastAPI server"""
        logger.info("🚀 Starting FastAPI server...")
        
        try:
            # Start server with uvicorn
            self.server_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "fastapi_rag_backend:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
            )
            
            # Wait for server to start with better detection
            logger.info("Waiting for server to start...")
            max_attempts = 30  # 30 seconds
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    # Check if process is still running
                    if self.server_process.poll() is not None:
                        # Process died, read output
                        output = self.server_process.stdout.read() if self.server_process.stdout else ""
                        logger.error(f"❌ FastAPI server process died: {output}")
                        return False
                    
                    # Try to connect to the server
                    with httpx.Client(timeout=2.0) as client:
                        response = client.get("http://localhost:8000/health")
                        if response.status_code == 200:
                            logger.info("✅ FastAPI server started successfully!")
                            return True
                        
                except httpx.RequestError:
                    # Server not ready yet, continue waiting
                    pass
                except Exception as e:
                    logger.debug(f"Connection attempt {attempt}: {e}")
                
                time.sleep(1)
                attempt += 1
            
            logger.error("❌ FastAPI server failed to start within 30 seconds")
            
            # Try to get some output from the process
            if self.server_process and self.server_process.stdout:
                try:
                    # Non-blocking read
                    import select
                    if select.select([self.server_process.stdout], [], [], 0.1)[0]:
                        output = self.server_process.stdout.readline()
                        logger.error(f"Server output: {output}")
                except:
                    pass
            
            return False
                
        except Exception as e:
            logger.error(f"❌ Error starting server: {e}")
            return False
    
    def test_api(self):
        """Quick API test with better error reporting"""
        logger.info("Testing API...")
        
        try:
            with httpx.Client(timeout=10.0) as client:
                # Test health endpoint
                response = client.get("http://localhost:8000/health")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ API health check: {data['status']}")
                    
                    # Show service status
                    services = data.get('services', {})
                    for service, status in services.items():
                        icon = "✅" if status == "healthy" else "❌"
                        logger.info(f"   {icon} {service}: {status}")
                    
                    # If any service is unhealthy, show suggestions
                    if any(status != "healthy" for status in services.values()):
                        logger.warning("⚠️ Some services are unhealthy:")
                        if services.get('llm') != 'healthy':
                            logger.warning("   - LLM issue: Check Ollama model availability")
                        if services.get('database') != 'healthy':
                            logger.warning("   - Database issue: Check Supabase connection")
                        if services.get('embeddings') != 'healthy':
                            logger.warning("   - Embeddings issue: Check sentence-transformers installation")
                    
                    return True
                else:
                    logger.error(f"❌ API health check failed: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ API test failed: {e}")
            return False
    
    def show_info(self):
        """Show running information"""
        print("\n" + "="*60)
        print("🤖 INFINITEPAY AI CHATBOT IS RUNNING!")
        print("="*60)
        print("📍 API Server: http://localhost:8000")
        print("📚 Documentation: http://localhost:8000/docs")
        print("🔍 Interactive API: http://localhost:8000/redoc")
        print("\n🧪 QUICK TEST:")
        print("curl -X POST http://localhost:8000/chat \\")
        print('  -H "Authorization: Bearer test-key" \\')
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"message": "Como funciona o pagamento?"}\'')
        print("\n📊 ENDPOINTS:")
        print("  POST /chat    - Main chatbot endpoint")
        print("  GET  /search  - Search documents")
        print("  GET  /health  - Health check")
        print("  GET  /models  - List available models")
        print("\n🎯 TESTING:")
        print("  python api_test_script.py  - Run comprehensive tests")
        print("  python debug_ollama.py    - Debug Ollama issues")
        print("\n⌨️  CONTROL:")
        print("  Ctrl+C - Stop the chatbot")
        print("="*60)
    
    def monitor(self):
        """Monitor the server"""
        logger.info("Monitoring server... Press Ctrl+C to stop")
        
        try:
            while True:
                if self.server_process.poll() is not None:
                    logger.error("❌ Server process died unexpectedly")
                    # Try to get exit code and output
                    exit_code = self.server_process.returncode
                    logger.error(f"Exit code: {exit_code}")
                    break
                
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        finally:
            self.cleanup()
    
    def run(self):
        """Main run method"""
        logger.info("🚀 Starting InfinitePay AI Chatbot")
        
        # Verify setup
        if not self.check_setup():
            logger.error("💡 Run: python debug_ollama.py to check your setup")
            return False
        
        # Check external services
        if not self.check_services():
            logger.error("💡 Run: python debug_ollama.py to debug Ollama issues")
            return False
        
        # Start server
        if not self.start_server():
            logger.error("💡 Check the server logs above for specific errors")
            logger.error("💡 Try running manually: python -m uvicorn fastapi_rag_backend:app --host 0.0.0.0 --port 8000")
            return False
        
        # Test API
        if not self.test_api():
            logger.warning("⚠️ API test failed, but server is running")
            logger.warning("💡 Run: python debug_ollama.py to check issues")
        
        # Show info and monitor
        self.show_info()
        self.monitor()
        
        return True

def main():
    """Main entry point"""
    runner = ChatbotRunner()
    
    try:
        success = runner.run()
        if not success:
            logger.error("❌ Failed to start chatbot")
            logger.error("💡 Debug steps:")
            logger.error("   1. python debug_ollama.py")
            logger.error("   2. Check .env file has correct Supabase credentials")
            logger.error("   3. Run manually: python fastapi_rag_backend.py")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        runner.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()