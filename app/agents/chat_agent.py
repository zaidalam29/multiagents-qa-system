from typing import List, Dict, Any
import requests
from app.config import settings

class ChatAgent:
    def __init__(self):
        self.client = settings.get_openai_client()
        self.vector_store = None  # Will be set later
    
    def set_vector_store(self, vector_store):
        """Set vector store instance"""
        self.vector_store = vector_store
    
    def answer_from_documents(self, query: str) -> str:
        """Answer query from documents"""
        if not self.vector_store:
            return "Vector store not initialized."
        
        # Search similar documents
        results = self.vector_store.search(query, top_k=3)
        
        if not results:
            return "No relevant documents found."
        
        # Build context from results
        context = "\n\n".join([
            f"Document {i+1}: {r['content'][:500]}..."
            for i, r in enumerate(results)
        ])
        
        # Generate answer using OpenAI
        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based only on the context provided."}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def search_web(self, query: str) -> str:
        """Search web for information"""
        try:
            # Simple web search using DuckDuckGo HTML
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('Abstract'):
                return data['Abstract']
            elif data.get('RelatedTopics'):
                return data['RelatedTopics'][0]['Text'] if data['RelatedTopics'] else "No results found."
            else:
                return "No information found from web search."
        
        except Exception as e:
            return f"Web search failed: {str(e)}"