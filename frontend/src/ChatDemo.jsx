'use client'

import React, { useState, useRef, useEffect } from 'react';
import { Send, MessageCircle, Minimize2, Maximize2, Bot, User, Code, Zap, Shield } from 'lucide-react';

const ChatBot = ({ isMinimized, setIsMinimized, onClose }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Olá! Eu sou o assistente de IA da InfinitePay. Posso ajudar a responder perguntas sobre os serviços, integrações, preços e muito mais sobre a InfinitePay. Como posso ajudar você hoje?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: messages.length + 1,
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      console.log('Sending request to API...');
      // Replace with your actual FastAPI endpoint
      const response = await fetch('https://infinitepay-chatbot-backend-production.up.railway.app/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputText,
          conversation_id: 'demo-session'
        })
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);

      if (!response.ok) {
        throw new Error(`Failed to get response: ${response.status}`);
      }

      const data = await response.json();
      console.log('Response data:', data);
      
      const botMessage = {
        id: messages.length + 2,
        text: data.response || "Desculpe, não consegui processar essa solicitação. Por favor, tente novamente.",
        sender: 'bot',
        timestamp: new Date(),
        sources: data.sources || []
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: messages.length + 2,
        text: "Estou com problemas para conectar no momento. Por favor, verifique se o backend está rodando no localhost:8000 ou tente novamente mais tarde.",
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (isMinimized) {
    return (
      <div className="fixed bottom-6 right-6 z-50">
        <button
          onClick={() => setIsMinimized(false)}
          className="bg-gradient-to-r from-lime-400 to-purple-500 hover:from-green-600 hover:to-purple-700 text-white rounded-full p-4 shadow-lg transform hover:scale-105 transition-all duration-200 flex items-center gap-2"
        >
          <MessageCircle size={24} />
          <span className="hidden sm:inline font-medium">Conversar com IA</span>
        </button>
      </div>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 w-96 h-[600px] bg-white rounded-xl shadow-2xl border border-gray-200 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-lime-400 to-purple-500 text-white p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-white/20 rounded-full p-2">
            <Bot size={20} />
          </div>
          <div>
            <h3 className="font-semibold text-lg">InfinitePay AI</h3>
            <p className="text-green-100 text-sm">Powered by RAG + Llama</p>
          </div>
        </div>
        <button
          onClick={() => setIsMinimized(true)}
          className="text-purple-500 hover:text-purple-700 transition-colors p-1"
        >
          <Minimize2 size={20} />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex gap-3 max-w-[85%] ${message.sender === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                message.sender === 'user' 
                  ? 'bg-gradient-to-r from-lime-400 to-purple-500' 
                  : 'bg-gray-200'
              }`}>
                {message.sender === 'user' ? (
                  <User size={16} className="text-white" />
                ) : (
                  <Bot size={16} className="text-gray-600" />
                )}
              </div>
              <div className={`rounded-2xl px-4 py-3 ${
                message.sender === 'user'
                  ? 'bg-gradient-to-r from-lime-400 to-purple-500 text-white'
                  : 'bg-white text-gray-800 shadow-sm border border-gray-100'
              }`}>
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.text}</p>
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-gray-200">
                    <p className="text-xs text-gray-500 mb-1">Sources:</p>
                    {message.sources.map((source, idx) => (
                      <p key={idx} className="text-xs text-green-600 truncate">{source}</p>
                    ))}
                  </div>
                )}
                <p className={`text-xs mt-2 ${
                  message.sender === 'user' ? 'text-green-100' : 'text-gray-400'
                }`}>
                  {formatTime(message.timestamp)}
                </p>
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="flex gap-3 max-w-[85%]">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                <Bot size={16} className="text-gray-600" />
              </div>
              <div className="bg-white rounded-2xl px-4 py-3 shadow-sm border border-gray-100">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 bg-white border-t border-gray-200">
        <div className="flex gap-2">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Pergunte-me sobre a InfinitePay..."
            className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-lime-400 focus:border-transparent text-sm"
            rows={1}
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={!inputText.trim() || isLoading}
            className="bg-gradient-to-r from-lime-400 to-purple-500 hover:from-green-600 hover:to-purple-500 disabled:from-gray-300 disabled:to-gray-300 text-white rounded-xl px-4 py-3 transition-all duration-200 disabled:cursor-not-allowed flex items-center justify-center"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Demo de Assistente de IA • RAG + FastAPI + Ollama
        </p>
      </div>
    </div>
  );
};

const DemoPage = () => {
  const [showChatbot, setShowChatbot] = useState(false); // Changed to false
  const [isMinimized, setIsMinimized] = useState(false); // Added this state here

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-purple-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-r from-lime-400 to-purple-500 rounded-lg flex items-center justify-center">
                <Bot className="text-white" size={20} />
              </div>
              <span className="text-xl font-bold text-gray-900">InfinitePay AI Assistant</span>
            </div>
            <div className="flex items-center gap-6">
              <a href="#demo" className="text-gray-600 hover:text-gray-900 font-medium">Demo</a>
              <a href="#tech" className="text-gray-600 hover:text-gray-900 font-medium">Tecnologias</a>
              <a href="#integration" className="text-gray-600 hover:text-gray-900 font-medium">Integração</a>
              <a 
                href="https://github.com/gsiq8/infinitepay-chatbot" 
                target="_blank" 
                rel="noopener noreferrer"
                className="bg-gradient-to-r from-lime-400 to-purple-500 text-white px-6 py-2 rounded-lg font-medium hover:from-green-600 hover:to-purple-700 transition-all">
                
                Veja o código
              </a>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            InfinitePay 
            <span className="bg-gradient-to-r from-lime-400 to-purple-500 bg-clip-text text-transparent"> AI Chatbot</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
            Um assistente inteligente de atendimento ao cliente construído com RAG (Geração Aumentada por Recuperação), 
            backend FastAPI e Ollama rodando Llama 3 localmente. Experimente a demonstração ao vivo abaixo!
          </p>
          <div className="flex justify-center gap-4">
            <button 
              onClick={() => {
                setShowChatbot(true);
                setIsMinimized(false);
              }}
              className="bg-gradient-to-r from-lime-400 to-purple-500 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:from-green-600 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg"
            >
              Teste Live Demo
            </button>
            <a 
              href="https://github.com/gsiq8/infinitepay-chatbot" 
              target="_blank" 
              rel="noopener noreferrer"
              className="bg-white text-gray-700 px-8 py-4 rounded-xl font-semibold text-lg border-2 border-gray-200 hover:border-gray-300 transition-all">
              Veja Code Fonte
            </a>
          </div>
        </div>

        {/* Tech Stack */}
        <div id="tech" className="mt-20">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">Tecnologias</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100 hover:shadow-xl transition-shadow">
              <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mb-6">
                <Zap className="w-6 h-6 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">RAG Backend</h3>
              <p className="text-gray-600 mb-4">FastAPI + LangChain + Supabase pgvector para recuperação de documentos inteligente</p>
              <div className="text-sm text-gray-500">
                <span className="bg-gray-100 px-2 py-1 rounded mr-2">FastAPI</span>
                <span className="bg-gray-100 px-2 py-1 rounded mr-2">Ollama</span>
                <span className="bg-gray-100 px-2 py-1 rounded">Llama 3</span>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100 hover:shadow-xl transition-shadow">
              <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mb-6">
                <Code className="w-6 h-6 text-purple-500" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Frontend Moderno</h3>
              <p className="text-gray-600 mb-4">React + Vite com suporte para widget incorporável e comunicação via iframe</p>
              <div className="text-sm text-gray-500">
                <span className="bg-gray-100 px-2 py-1 rounded mr-2">React</span>
                <span className="bg-gray-100 px-2 py-1 rounded mr-2">Vite</span>
                <span className="bg-gray-100 px-2 py-1 rounded">Tailwind</span>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100 hover:shadow-xl transition-shadow">
              <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mb-6">
                <Shield className="w-6 h-6 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Pronto para Produção</h3>
              <p className="text-gray-600 mb-4">Filtros de segurança, limitação de taxa e otimizado para implantação em alto tráfego</p>
              <div className="text-sm text-gray-500">
                <span className="bg-gray-100 px-2 py-1 rounded mr-2">Segurança</span>
                <span className="bg-gray-100 px-2 py-1 rounded mr-2">Limitação de Taxa</span>
                <span className="bg-gray-100 px-2 py-1 rounded">Otimizado</span>
              </div>
            </div>
          </div>
        </div>

        {/* Demo Section */}
        <div id="demo" className="mt-20 text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Demo ao Vivo</h2>
          <p className="text-lg text-gray-600 mb-8">
            O assistente de IA foi treinado com o conteúdo do site da InfinitePay e pode responder perguntas sobre:
          </p>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h4 className="font-semibold text-gray-900">Soluções de Pagamento</h4>
              <p className="text-sm text-gray-600">Soluções de pagamento, maquininhas de cartão, carteiras digitais</p>
            </div>
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h4 className="font-semibold text-gray-900">Preços & Taxas</h4>
              <p className="text-sm text-gray-600">Taxas de transação, planos, estruturas de taxas</p>
            </div>
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h4 className="font-semibold text-gray-900">Integração</h4>
              <p className="text-sm text-gray-600">APIs, SDKs, documentação técnica</p>
            </div>
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h4 className="font-semibold text-gray-900">Suporte</h4>
              <p className="text-sm text-gray-600">Suporte, soluções, configurações e ajuda com a sua conta</p>
            </div>
          </div>
          {!showChatbot && (
            <button 
              onClick={() => {
                setShowChatbot(true);
                setIsMinimized(false);
              }}
              className="bg-gradient-to-r from-lime-400 to-purple-500 text-white px-8 py-3 rounded-lg font-semibold hover:from-green-600 hover:to-purple-700 transition-all"
            >
              Abrir Assistente de IA
            </button>
          )}
        </div>

        {/* Integration Guide */}
        <div id="integration" className="mt-20">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">Integração Rápida</h2>
          <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100">
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Incorporar Widget</h3>
                <p className="text-gray-600 mb-4">Adicione o assistente de IA a qualquer site com um iframe simples:</p>
                <div className="bg-gray-100 rounded-lg p-4 text-black font-mono">
                  {`<iframe 
  src="https://your-domain.com/widget" 
  width="400" 
  height="600"
  frameborder="0">
</iframe>`}
                </div>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Integração com API</h3>
                <p className="text-gray-600 mb-4">Conecte diretamente ao backend FastAPI:</p>
                <div className="bg-gray-100 rounded-lg p-4 text-black font-mono">
                  {`POST /chat
{
  "query": "Como faço para integrar pagamentos?",
  "conversation_id": "user-123"
}`}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Chatbot - Pass props to control state */}
      {showChatbot && (
        <ChatBot 
          isMinimized={isMinimized}
          setIsMinimized={setIsMinimized}
          onClose={() => setShowChatbot(false)}
        />
      )}
    </div>
  );
};

export default DemoPage;