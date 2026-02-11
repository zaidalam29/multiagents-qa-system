from fastapi import FastAPI, Depends, HTTPException, status, Security, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
from typing import List, Optional
import tempfile
import os
from pydantic import BaseModel, Field

from app.config import settings
from app.vector_store import VectorStore
from app.file_processor import FileProcessor
from app.agents.orchestrator import OrchestratorAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.research_agent import ResearchAgent
from app.agents.validation_agent import ValidationAgent
from app.utils.guardrails import sanitize_input
import time
from fastapi import Request
# API Key Security
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)

# Initialize components
vector_store = VectorStore()
retrieval_agent = RetrievalAgent(vector_store)
research_agent = ResearchAgent()
validation_agent = ValidationAgent()
orchestrator = OrchestratorAgent(retrieval_agent, research_agent, validation_agent)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    # Startup
    print("=" * 50)
    print("PDF QA System Starting Up...")
    print(f"LLM Provider: {orchestrator.provider}")
    print(f"Vector Store: {settings.VECTOR_DB_PATH}")
    print("=" * 50)
    
    # Create vector store directory
    os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
    
    yield
    
    # Shutdown
    print("\n" + "=" * 50)
    print("PDF QA System Shutting Down...")
    print("=" * 50)

app = FastAPI(
    title="PDF QA System with Multi-Agent Architecture",
    description="Advanced QA system with document retrieval and web research",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security dependency
async def get_api_key(api_key: str = Security(api_key_header)):
    """API key validation"""
    if settings.DEBUG:
        # Allow all in debug mode
        return "debug_key"
    
    # In production, implement proper API key validation
    # For now, allow if API key is provided
    if api_key:
        return api_key
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key"
    )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    # Skip logging for health checks
    if request.url.path not in ["/health", "/"]:
        print(f"\n[{time.strftime('%H:%M:%S')}] {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    if process_time > 2.0:  # Log slow requests
        print(f"[SLOW] Process time: {process_time:.2f}s for {request.url.path}")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response    

# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    collection_name: Optional[str] = "default"
    use_internet: Optional[bool] = True
    top_k: Optional[int] = Field(5, ge=1, le=20)

class QueryResponse(BaseModel):
    answer: str
    agents_used: List[str]
    processing_time: float
    has_internet_research: bool

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    chunks_processed: int
    collection: str
    document_ids: List[str]

# Upload endpoint
@app.post("/api/v1/upload/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = "default",
    api_key: str = Depends(get_api_key)
):
    """Upload and process PDF/TXT file"""
    
    # Simple file type validation
    allowed_extensions = ['.pdf', '.txt', '.md']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed: {allowed_extensions}"
        )
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        
        # Validate file size (10MB max)
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=400,
                detail="File too large. Max size: 10MB"
            )
        
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Process file
        file_type = file_ext[1:]  # Remove dot
        chunks = FileProcessor.process_file(tmp_path, file_type)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid content found in file")
        
        # Prepare for vector store
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Add to vector store
        ids = vector_store.add_documents(
            documents=documents,
            metadatas=metadatas,
            collection_name=collection_name
        )
        
        # Get collection stats
        stats = vector_store.get_collection_stats(collection_name)
        
        return UploadResponse(
            status="success",
            message=f"File processed successfully",
            filename=file.filename,
            chunks_processed=len(chunks),
            collection=collection_name,
            document_ids=ids[:5]  # Return first 5 IDs
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Query endpoint
@app.post("/api/v1/query/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    """Query documents and get answer"""
    
    import time
    start_time = time.time()
    
    # Sanitize input
    sanitized_query = sanitize_input(request.query)
    
    if not sanitized_query:
        raise HTTPException(status_code=400, detail="Invalid query")
    
    # Prepare context
    context = {
        "collection_name": request.collection_name,
        "top_k": request.top_k,
        "use_internet": request.use_internet
    }
    
    try:
        # Process query through orchestrator
        result = orchestrator.process_query(sanitized_query, context)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=result["answer"],
            agents_used=result["agents_used"],
            processing_time=round(processing_time, 2),
            has_internet_research="research" in result["agents_used"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# List collections endpoint
@app.get("/api/v1/collections/")
async def list_collections(api_key: str = Depends(get_api_key)):
    """List all collections"""
    collections = vector_store.list_collections()
    
    # Get stats for each collection
    collections_with_stats = []
    for collection in collections:
        stats = vector_store.get_collection_stats(collection)
        collections_with_stats.append(stats)
    
    return {
        "collections": collections_with_stats
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    collections = vector_store.list_collections()
    
    return {
        "status": "healthy",
        "service": "PDF QA System",
        "version": "2.0.0",
        "llm_provider": orchestrator.provider,
        "collections_count": len(collections),
        "collections": collections
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF QA System with Multi-Agent Architecture",
        "version": "2.0.0",
        "endpoints": {
            "upload": "POST /api/v1/upload/",
            "query": "POST /api/v1/query/",
            "collections": "GET /api/v1/collections/",
            "health": "GET /health",
            "docs": "/docs" if settings.DEBUG else "disabled in production"
        },
        "features": [
            "PDF/TXT file upload",
            "Vector similarity search",
            "Multi-agent architecture",
            "OpenAI + OpenRouter support",
            "Web research capability"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.APP_HOST, 
        port=settings.APP_PORT,
        reload=settings.DEBUG
    )