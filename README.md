<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
</p>

<h1 align="center">🤖 AI-Powered Document Question Answering System</h1>

<h3 align="center">
Multi-Agent Architecture • Vector Search • Web Research • Production Ready
</h3>

<p align="center">
<b>Zaid Alam</b><br>
Full Stack Developer | Gen AI/ML Engineer | RAG | Agentic AI
</p>


------------------------------------------------------------------------

## 📋 Table of Contents

-   [✨ Features](#-features)
-   [🏗️ Architecture](#️-architecture)
-   [🚀 Quick Start](#-quick-start)
-   [📚 API Endpoints](#-api-endpoints)
-   [🤖 Multi-Agent System](#-multi-agent-system)
-   [🔧 Configuration](#-configuration)
-   [📁 Project Structure](#-project-structure)
-   [🧪 Testing](#-testing)
-   [📊 Performance](#-performance)
-   [🔒 Security](#-security)
-   [🐳 Docker Deployment](#-docker-deployment)
-   [❓ Troubleshooting](#-troubleshooting)
-   [📄 License](#-license)

------------------------------------------------------------------------

## ✨ Features

### 🎯 Core Capabilities

| Feature                | Description                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| 📄 Document Processing | Upload and process PDF, TXT, MD files with intelligent chunking (1000 chars with 200 overlap) |
| 🧠 Vector Search       | Semantic search using OpenAI embeddings or local Sentence Transformers                        |
| 🤖 Multi-Agent System  | 4 specialized AI agents working in concert                                                    |
| 🌐 Web Research        | DuckDuckGo integration for real-time information                                              |
| ⚡ Fast Responses       | LRU caching, batch processing, connection pooling                                             |
| 🔒 Enterprise Security | API key authentication, input sanitization, guardrails                                        |
| 🔄 LLM Fallback        | Automatic failover from OpenAI to OpenRouter                                                  |
| 📊 Collections         | Organize documents into collections                                                           |
| 📈 Scalable            | SQLite, FAISS, or ChromaDB support                                                            |

---

### 🛠 Tech Stack

| Component     | Technology                      | Version  |
| ------------- | ------------------------------- | -------- |
| Framework     | FastAPI                         | 0.104+   |
| Language      | Python                          | 3.10+    |
| Vector DB     | SQLite + FAISS                  | Built-in |
| Embeddings    | OpenAI / Sentence Transformers  | -        |
| LLM           | GPT-4 / Claude / Gemini / Llama | -        |
| Web Search    | DuckDuckGo                      | -        |
| Deployment    | Docker, Uvicorn                 | -        |
| Documentation | Swagger UI, ReDoc               | Built-in |

---


## 🏗️ Architecture

``` mermaid
graph TB
    A[User] --> B[FastAPI Server]
    B --> C[Router Layer]
    C --> D[Upload Endpoint]
    C --> E[Query Endpoint]

    E --> F[Orchestrator Agent]
    F --> G[Retrieval Agent]
    F --> H[Research Agent]
    F --> I[Validation Agent]

    G --> J[(Vector DB)]
    H --> K[DuckDuckGo]
    I --> L[OpenAI/OpenRouter]
```

### Query Flow

    Client → FastAPI → Orchestrator → Agent Selection
            ↓
    Retrieval Agent → Vector Store
            ↓
    Research Agent → Web Search
            ↓
    Validation Agent → LLM
            ↓
    Final Answer → Client

------------------------------------------------------------------------

## 🚀 Quick Start

### 📦 One-Line Installation

``` bash
git clone https://github.com/zaidalam29/multiagents-qa-system.git && \
cd multiagents-qa-system && \
python -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt
```

------------------------------------------------------------------------

### 📋 Step-by-Step Installation

``` bash
git clone https://github.com/zaidalaam29/multiagents-qa-system.git
cd multiagents-qa-system

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

pip install sentence-transformers

cp .env.example .env

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

------------------------------------------------------------------------

## 📄 requirements.txt

``` txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic-settings==2.1.0
pydantic==2.5.0
python-dotenv==1.0.0
python-multipart==0.0.6
openai==1.3.8
PyPDF2==3.0.1
sentence-transformers==2.2.2
numpy==1.24.3
httpx==0.25.0
tiktoken==0.5.2
```

------------------------------------------------------------------------

## 🔐 .env.example

``` env
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small

OPENROUTER_API_KEY=sk-or-your-openrouter-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=anthropic/claude-3-opus

USE_LOCAL_EMBEDDINGS=False
REQUEST_TIMEOUT=30
ENABLE_CACHING=True

APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=True

API_KEY_HEADER=X-API-Key
VALID_API_KEYS=debug_key,test_key
MAX_FILE_SIZE=10485760
ALLOWED_FILE_TYPES=.pdf,.txt,.md

VECTOR_DB_PATH=./vector_store
VECTOR_DB_TYPE=sqlite
```

------------------------------------------------------------------------

## 📚 API Endpoints

Base URL: `http://localhost:8000`

| Method | Endpoint             | Description      | Auth |
| ------ | -------------------- | ---------------- | ---- |
| POST   | /api/v1/upload/      | Upload document  | ✅    |
| POST   | /api/v1/query/       | Ask question     | ✅    |
| GET    | /api/v1/collections/ | List collections | ✅    |
| GET    | /health              | Health check     | ❌    |
| GET    | /docs                | Swagger UI       | ❌    |
| GET    | /redoc               | ReDoc            | ❌    |


------------------------------------------------------------------------

## 🤖 Multi-Agent System

### 🎭 Agent Architecture

    ORCHESTRATOR AGENT
      ├── Retrieval Agent (Vector Search)
      ├── Research Agent (Web Search)
      └── Validation Agent (LLM Validation)

### 🎯 Agent Decision Matrix

| Query Type         | Retrieval | Research | Validation |
| ------------------ | --------- | -------- | ---------- |
| Document-based     | ✅         | ❌        | ✅          |
| Latest information | ❌         | ✅        | ✅          |
| Comparison queries | ✅         | ✅        | ✅          |

---

## 🔧 Configuration

### ⚙️ Performance Optimization Matrix

| Configuration     | Speed     | Accuracy  | Cost |
| ----------------- | --------- | --------- | ---- |
| OpenAI Embeddings | Slow      | Excellent | High |
| Local Embeddings  | Fast      | Good      | Free |
| LRU Cache         | Very Fast | Same      | Same |
| SQLite            | Fast      | N/A       | Free |
| FAISS             | Faster    | N/A       | Free |

---


## 📁 Project Structure

```
multiagents-qa-system/
│
├── 📁 app/
│   ├── 📄 __init__.py
│   ├── 📄 main.py                 # FastAPI application (endpoints, middleware, lifespan)
│   ├── 📄 config.py               # Configuration management (env vars, settings)
│   ├── 📄 vector_store.py         # Vector database operations (SQLite/FAISS)
│   ├── 📄 file_processor.py       # PDF/TXT processing (chunking, text extraction)
│   │
│   ├── 📁 agents/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 orchestrator.py     # Main orchestrator agent (decision making)
│   │   ├── 📄 retrieval_agent.py  # Document retrieval agent (vector search)
│   │   ├── 📄 research_agent.py   # Web research agent (DuckDuckGo)
│   │   └── 📄 validation_agent.py # Answer validation agent (quality check)
│   │
│   └── 📁 utils/
│       ├── 📄 __init__.py
│       ├── 📄 guardrails.py       # Content validation (toxicity, PII, harmful)
│       └── 📄 security.py         # API key authentication
│
├── 📁 vector_store/               # SQLite database files
│   └── 📄 vectors.db              # Main database (created automatically)
│
├── 📁 docs/                       # Additional documentation
│   └── 📄 architecture.md
│
├── 📄 .env                        # Environment variables (create from .env.example)
├── 📄 .env.example                # Example environment variables
├── 📄 requirements.txt            # Python dependencies
├── 📄 Dockerfile                  # Docker configuration
├── 📄 docker-compose.yml          # Docker compose configuration
├── 📄 README.md                   # This file
├── 📄 LICENSE                     # MIT License
└── 📄 .gitignore                  # Git ignore rules
```


------------------------------------------------------------------------

## 📊 Performance

-   Batch embedding generation
-   LRU caching for repeated queries
-   Connection pooling
-   Optimized chunk retrieval

------------------------------------------------------------------------

## 🔒 Security

-   API key authentication
-   File size validation
-   Input sanitization
-   Guardrails against harmful input
-   PII filtering

------------------------------------------------------------------------

## 🐳 Docker Deployment

``` bash
docker build -t multiagents-qa-system .
docker run -p 8000:8000 multiagents-qa-system
```

------------------------------------------------------------------------
## ❓ Troubleshooting

| Issue          | Solution              |
| -------------- | --------------------- |
| OpenAI errors  | Verify API key        |
| Slow responses | Enable caching        |
| Upload failure | Check file size limit |

---

## 📄 License

MIT License © 2026

**Zaid Alam**
Full Stack Developer • Gen AI/ML Engineer • RAG • Agentic AI Engineer

🌐 https://zaidalam.in
