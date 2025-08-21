import React from 'react'
import { createRoot } from 'react-dom/client'

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
        // Replace with your actual FastAPI endpoint
        const response = await fetch('https://giovanasiquieroli.com.br', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: inputText,
            conversation_id: 'demo-session'
          })
        });
  
        if (!response.ok) {
          throw new Error('Failed to get response');
        }
  
        const data = await response.json();
        
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

// Widget initialization
window.InfinitePayChatbot = {
  init: (config = {}) => {
    const container = document.createElement('div')
    container.id = 'infinitepay-chatbot-widget'
    document.body.appendChild(container)

    const root = createRoot(container)
    root.render(<ChatBot {...config} />)
  }
}

// Auto-initialize if script has data-auto-init
const script = document.currentScript
if (script && script.hasAttribute('data-auto-init')) {
  window.InfinitePayChatbot.init()
}