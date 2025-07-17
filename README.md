# Modular AI Chatbot

A flexible and modular chatbot implementation that supports multiple AI providers (Azure OpenAI, Google AI, Ollama) with RAG (Retrieval Augmented Generation) capabilities and vector database integration.

Last Updated: 2025-07-14 13:51:11  
Current User: animeshpatni94

## Purpose

This project provides a modular chatbot solution with the following key features:
- Support for multiple AI providers (Azure OpenAI, Google AI, Ollama) through a provider layer abstraction
- Vector database integration (Azure Search, Qdrant) for efficient similarity search
- Real-time streaming responses with markdown support
- Session management for conversation context
- RAG (Retrieval Augmented Generation) capabilities for enhanced responses
- Clean and responsive web interface
- Citation support for referenced documents

## Technical Architecture

### Components
1. **Provider Layer**
   - Base Provider (`base_llm_provider.py`): Abstract interface for LLM implementations
   - Provider Implementations:
     - Azure OpenAI (`azureai_layer.py`)
     - Google AI (`googleai_layer.py`)
     - Ollama (`ollama_layer.py`)

2. **Vector Database Layer**
   - Base Provider (`base_vectordb_provider.py`)
   - Vector Store Manager (`vectordb_manager.py`)
   - Implementations:
     - Qdrant (`qdrant_vectordb_layer.py`)
     - Azure Search (`azuresearch_vectordb_provider.py`)

3. **API Layer** (`api.py`)
   - Flask-based RESTful API
   - WebSocket support for real-time streaming
   - Session management using UUID-based tracking
   - RAG pipeline integration

4. **Embedding Layer**
   - Base Embedding Provider (`base_embedding_provider.py`)
   - Implementations:
     - Azure AI (`azureai_embedding_layer.py`)
     - Ollama (`ollama_embedding_layer.py`)

5. **Text Processing**
   - Text Ingestor (`text_ingestor.py`) supporting:
     - C# code (.cs)
     - SQL files (.sql)
     - XML files (.xml)
     - General text files

### Configuration (`config.ini`)
```ini
[AZUREAI]
azure_deployment=your-deployment
api_version=2023-05-15
azure_endpoint=https://your-endpoint.openai.azure.com/
openai_api_key=your-key

[GOOGLEAI]
model_name=gemini-pro
google_api_key=your-key

[OLLAMA]
model=llama2
base_url=http://localhost:11434

[OLLAMA_EMBED]
model=llama2
base_url=http://localhost:11434

[QDRANT]
url=http://localhost:6333
collection_name=your_collection

[AZURESEARCH]
endpoint=your-endpoint
api_key=your-key
index_name=your-index
```

## Development Environment Setup

1. Clone the repository
```bash
git clone https://github.com/animeshpatni94/Modular-AI-Chatbot.git
cd Modular-AI-Chatbot
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run development server
```bash
python python -m Web.api
```

## Performance Considerations

1. **Vector Database**
   - Custom chunking strategies for different file types
   - Intelligent text splitting with overlap
   - Support for metadata filtering
   - Configurable retrieval settings (k=50 default)

2. **Ollama Integration**
   - Configurable context window (default: 3000)
   - Temperature setting: 0.7
   - Support for both chat and embedding models

3. **Session Management**
   - Default history size: 6 messages
   - Memory-based storage
   - UUID-based session tracking

## Contributing

1. Fork the repository
2. Create a feature branch
```bash
git checkout -b feature/your-feature-name
```
3. Implement changes and tests
4. Submit a pull request