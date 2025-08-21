#!/usr/bin/env python3
"""
Fix data and model issues for InfinitePay AI Chatbot
Addresses both the 404 Ollama error and empty document titles
"""

import os
import sys
import json
import httpx
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

class DataAndModelFixer:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.supabase_client = None
        self.working_model = None
    
    async def fix_ollama_model(self):
        """Find and test working Ollama model"""
        print("🔧 Fixing Ollama model configuration...")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get available models
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code != 200:
                    print("❌ Ollama not responding")
                    return False
                
                data = response.json()
                models = [m['name'] for m in data.get('models', [])]
                
                if not models:
                    print("❌ No Ollama models found")
                    print("💡 Install a model: ollama pull llama3.2")
                    return False
                
                print(f"📋 Available models: {models}")
                
                # Test each model to find one that works
                for model_name in models:
                    print(f"🧪 Testing model: {model_name}")
                    
                    try:
                        test_response = await client.post(
                            f"{self.ollama_url}/api/generate",
                            json={
                                "model": model_name,
                                "prompt": "Hello",
                                "stream": False
                            },
                            timeout=30.0
                        )
                        
                        if test_response.status_code == 200:
                            result = test_response.json()
                            if result.get('response'):
                                print(f"✅ Model {model_name} works!")
                                self.working_model = model_name
                                
                                # Update .env file
                                self.update_env_file(model_name)
                                return True
                        else:
                            print(f"❌ Model {model_name} failed: {test_response.status_code}")
                            
                    except Exception as e:
                        print(f"❌ Error testing {model_name}: {e}")
                        continue
                
                print("❌ No working models found")
                return False
                
        except Exception as e:
            print(f"❌ Error checking Ollama: {e}")
            return False
    
    def update_env_file(self, model_name):
        """Update .env file with working model"""
        try:
            env_path = Path('.env')
            if env_path.exists():
                with open(env_path, 'r') as f:
                    content = f.read()
                
                # Update or add OLLAMA_MODEL
                lines = content.split('\n')
                updated = False
                
                for i, line in enumerate(lines):
                    if line.startswith('OLLAMA_MODEL='):
                        lines[i] = f'OLLAMA_MODEL={model_name}'
                        updated = True
                        break
                
                if not updated:
                    lines.append(f'OLLAMA_MODEL={model_name}')
                
                with open(env_path, 'w') as f:
                    f.write('\n'.join(lines))
                
                print(f"✅ Updated .env with OLLAMA_MODEL={model_name}")
            else:
                # Create new .env file
                with open(env_path, 'w') as f:
                    f.write(f'OLLAMA_MODEL={model_name}\n')
                print(f"✅ Created .env with OLLAMA_MODEL={model_name}")
                
        except Exception as e:
            print(f"⚠️ Could not update .env file: {e}")
            print(f"💡 Please manually add: OLLAMA_MODEL={model_name}")
    
    def init_supabase(self):
        """Initialize Supabase client"""
        try:
            supabase_url = os.getenv('SUPABASE_URL')
            service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            anon_key = os.getenv('SUPABASE_ANON_KEY')
            
            if not supabase_url:
                print("❌ SUPABASE_URL not found in environment")
                return False
            
            # Prefer service role key for admin operations
            if service_key:
                self.supabase_client = create_client(supabase_url, service_key)
                print("✅ Using Supabase service role key")
            elif anon_key:
                self.supabase_client = create_client(supabase_url, anon_key)
                print("✅ Using Supabase anon key")
            else:
                print("❌ No Supabase key found")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing Supabase: {e}")
            return False
    
    def check_document_data(self):
        """Check document data quality"""
        print("\n🔍 Checking document data quality...")
        
        try:
            # Get sample documents
            result = self.supabase_client.table('documents').select('*').limit(10).execute()
            
            if not result.data:
                print("❌ No documents found in database")
                return False
            
            docs = result.data
            print(f"📊 Found {len(docs)} sample documents")
            
            # Analyze data quality
            empty_titles = 0
            empty_content = 0
            short_content = 0
            
            for doc in docs:
                title = doc.get('page_title', '')
                content = doc.get('content', '')
                
                if not title or title.strip() == '':
                    empty_titles += 1
                
                if not content or content.strip() == '':
                    empty_content += 1
                elif len(content.strip()) < 50:
                    short_content += 1
            
            print(f"📈 Data quality analysis:")
            print(f"   Empty titles: {empty_titles}/{len(docs)}")
            print(f"   Empty content: {empty_content}/{len(docs)}")
            print(f"   Short content (<50 chars): {short_content}/{len(docs)}")
            
            # Show sample document
            if docs:
                sample = docs[0]
                print(f"\n📄 Sample document:")
                print(f"   Title: '{sample.get('page_title', 'EMPTY')}' ")
                print(f"   Content: '{sample.get('content', 'EMPTY')[:100]}...'")
                print(f"   URL: '{sample.get('page_url', 'EMPTY')}'")
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking document data: {e}")
            return False
    
    def fix_document_titles(self):
        """Fix empty document titles"""
        print("\n🔧 Fixing empty document titles...")
        
        try:
            # Get documents with empty titles
            result = self.supabase_client.table('documents').select('*').is_('page_title', None).execute()
            empty_title_docs = result.data or []
            
            # Also check for empty string titles
            result2 = self.supabase_client.table('documents').select('*').eq('page_title', '').execute()
            empty_title_docs.extend(result2.data or [])
            
            if not empty_title_docs:
                print("✅ No documents with empty titles found")
                return True
            
            print(f"🔧 Fixing {len(empty_title_docs)} documents with empty titles...")
            
            fixed_count = 0
            for doc in empty_title_docs:
                try:
                    doc_id = doc['id']
                    content = doc.get('content', '')
                    
                    # Generate title from content
                    if content:
                        # Use first sentence or first 50 characters as title
                        sentences = content.split('.')
                        if sentences and len(sentences[0]) > 5:
                            new_title = sentences[0].strip()[:100]
                        else:
                            new_title = content[:50].strip()
                        
                        if len(new_title) < 5:
                            new_title = f"Documento {doc_id}"
                    else:
                        new_title = f"Documento {doc_id}"
                    
                    # Update document
                    update_result = self.supabase_client.table('documents').update({
                        'page_title': new_title
                    }).eq('id', doc_id).execute()
                    
                    if update_result.data:
                        fixed_count += 1
                        if fixed_count <= 3:  # Show first 3 fixes
                            print(f"   ✅ Fixed doc {doc_id}: '{new_title}'")
                    
                except Exception as e:
                    print(f"   ❌ Error fixing doc {doc.get('id', 'unknown')}: {e}")
            
            print(f"✅ Fixed {fixed_count} document titles")
            return True
            
        except Exception as e:
            print(f"❌ Error fixing document titles: {e}")
            return False
    
    def create_sample_data(self):
        """Create sample data if database is empty"""
        print("\n🏗️ Creating sample data...")
        
        sample_docs = [
            {
                "content": "O InfinitePay oferece soluções completas de pagamento digital. Com nossa maquininha você pode aceitar pagamentos via cartão de crédito, débito e Pix. As taxas são competitivas e o atendimento é personalizado.",
                "metadata": {"source": "homepage", "section": "produtos"},
                "page_title": "InfinitePay - Soluções de Pagamento",
                "page_url": "https://infinitepay.io/produtos",
                "chunk_index": 0,
                "is_public": True
            },
            {
                "content": "O pagamento via Pix é instantâneo e funciona 24 horas por dia, 7 dias por semana. Com nossa solução, você recebe o dinheiro na hora e pode confirmar as vendas imediatamente. Taxa zero para recebimentos via Pix.",
                "metadata": {"source": "pix", "section": "beneficios"},
                "page_title": "Pix InfinitePay - Receba na Hora",
                "page_url": "https://infinitepay.io/pix",
                "chunk_index": 0,
                "is_public": True
            },
            {
                "content": "Para criar uma conta MEI com o InfinitePay é muito simples. Você precisa do seu CPF, dados pessoais e informações sobre sua atividade. O processo é 100% digital e pode ser feito em poucos minutos.",
                "metadata": {"source": "mei", "section": "como-fazer"},
                "page_title": "Como Criar Conta MEI",
                "page_url": "https://infinitepay.io/mei",
                "chunk_index": 0,
                "is_public": True
            },
            {
                "content": "As taxas da maquininha InfinitePay começam em 1,99% para débito e 2,99% para crédito à vista. Para crédito parcelado, as taxas variam conforme o número de parcelas. Sem taxa de adesão ou mensalidade.",
                "metadata": {"source": "taxas", "section": "precos"},
                "page_title": "Taxas e Preços InfinitePay",
                "page_url": "https://infinitepay.io/taxas",
                "chunk_index": 0,
                "is_public": True
            },
            {
                "content": "O follow up de vendas é fundamental para aumentar sua conversão. Com nosso sistema, você pode enviar lembretes automáticos, links de pagamento personalizados e acompanhar o status de cada transação em tempo real.",
                "metadata": {"source": "vendas", "section": "follow-up"},
                "page_title": "Follow Up de Vendas",
                "page_url": "https://infinitepay.io/vendas",
                "chunk_index": 0,
                "is_public": True
            }
        ]
        
        try:
            # Check if we need embeddings
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Generate embeddings for sample docs
            texts = [doc['content'] for doc in sample_docs]
            embeddings = model.encode(texts)
            
            # Add embeddings to documents
            for i, doc in enumerate(sample_docs):
                doc['embedding'] = embeddings[i].tolist()
            
            # Insert documents
            result = self.supabase_client.table('documents').insert(sample_docs).execute()
            
            if result.data:
                print(f"✅ Created {len(result.data)} sample documents")
                return True
            else:
                print("❌ Failed to create sample documents")
                return False
                
        except Exception as e:
            print(f"❌ Error creating sample data: {e}")
            return False
    
    async def test_end_to_end(self):
        """Test the complete chat flow"""
        print("\n🧪 Testing end-to-end chat flow...")
        
        if not self.working_model:
            print("❌ No working model found")
            return False
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Test the chat endpoint
                response = await client.post(
                    "http://localhost:8000/chat",
                    headers={"Authorization": "Bearer test-key"},
                    json={"message": "Como funciona o Pix?"}
                )
                
                print(f"Chat response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Chat working!")
                    print(f"Response: {data.get('response', 'No response')[:200]}...")
                    print(f"Sources: {len(data.get('sources', []))}")
                    return True
                else:
                    print(f"❌ Chat failed: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"❌ End-to-end test failed: {e}")
            return False
    
    async def run_fixes(self):
        """Run all fixes"""
        print("🔧 InfinitePay AI Chatbot - Data & Model Fixer")
        print("=" * 60)
        
        # Step 1: Fix Ollama model
        if not await self.fix_ollama_model():
            print("❌ Failed to fix Ollama model")
            return False
        
        # Step 2: Initialize Supabase
        if not self.init_supabase():
            print("❌ Failed to initialize Supabase")
            return False
        
        # Step 3: Check document data
        if not self.check_document_data():
            # Try to create sample data
            print("💡 Creating sample data...")
            if not self.create_sample_data():
                print("❌ Failed to create sample data")
                return False
        
        # Step 4: Fix document titles
        self.fix_document_titles()
        
        # Step 5: Test end-to-end (optional, requires server to be running)
        print("\n" + "=" * 60)
        print("✅ All fixes completed!")
        print("\n💡 Next steps:")
        print("1. Restart your FastAPI server:")
        print("   python fixed_run_chatbot.py")
        print("2. Test the API:")
        print("   python api_test_script.py")
        
        return True

async def main():
    fixer = DataAndModelFixer()
    success = await fixer.run_fixes()
    
    if success:
        print("\n🎉 All issues should be fixed!")
    else:
        print("\n❌ Some issues remain. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())