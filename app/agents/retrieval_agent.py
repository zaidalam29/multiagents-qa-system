from typing import Dict, Any, Optional
from app.config import settings

class RetrievalAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.client, self.model, self.provider = settings.get_llm_client()
    
    def execute(self, query: str, context: Optional[Dict] = None, 
                previous_results: Optional[Dict] = None) -> str:
        """Execute retrieval agent"""
        
        # Get search parameters from context
        collection_name = context.get('collection_name', 'default') if context else 'default'
        top_k = context.get('top_k', 5) if context else 5
        
        # Search in vector store
        search_results = self.vector_store.search(
            query=query,
            top_k=top_k,
            collection_name=collection_name
        )
        
        if not search_results:
            return "No relevant information found in the uploaded documents."
        
        # Format context from retrieved documents
        context_text = "\n\n".join([
            f"Document {i+1} (Source: {r.get('source', 'Unknown')}):\n{r['document'][:500]}"
            for i, r in enumerate(search_results[:3])
        ])
        
        # Generate answer based on retrieved context
        prompt = f"""You are a precise information retrieval agent. 
Answer the question based ONLY on the provided context.
If the context doesn't contain relevant information, say so clearly.

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You answer questions based only on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"