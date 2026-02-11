 
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional

from app.agents.orchestrator import OrchestratorAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.research_agent import ResearchAgent
from app.agents.validation_agent import ValidationAgent
from app.vector_store import VectorStore
from app.utils.guardrails import sanitize_input

router = APIRouter()

# Initialize agents
vector_store = VectorStore()
retrieval_agent = RetrievalAgent(vector_store)
research_agent = ResearchAgent()
validation_agent = ValidationAgent()
orchestrator = OrchestratorAgent(retrieval_agent, research_agent, validation_agent)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    collection_name: Optional[str] = Field("documents", description="Collection name")
    use_internet: Optional[bool] = Field(True, description="Allow internet research")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results to retrieve")

class QueryResponse(BaseModel):
    answer: str
    agents_used: list
    sources: list
    processing_time: float
    has_internet_research: bool

@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
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
        
        # Extract sources from retrieval results
        sources = []
        if "retrieval" in result.get("intermediate_results", {}):
            retrieval_context = result["intermediate_results"].get("retrieval", "")
            # Parse sources from retrieval context (simplified)
            sources = self._extract_sources(retrieval_context)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=result["answer"],
            agents_used=result["agents_used"],
            sources=sources,
            processing_time=round(processing_time, 2),
            has_internet_research="research" in result["agents_used"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    @staticmethod
    def _extract_sources(context: str) -> List[str]:
        """Extract source information from context"""
        import re
        sources = re.findall(r'Source:\s*([^\n]+)', context)
        return list(set(sources))  # Remove duplicates