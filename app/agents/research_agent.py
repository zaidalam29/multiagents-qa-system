from typing import Dict, Any, Optional
import requests
from app.config import settings

class ResearchAgent:
    def __init__(self):
        self.client, self.model, self.provider = settings.get_llm_client()
    
    def execute(self, query: str, context: Optional[Dict] = None, 
                previous_results: Optional[Dict] = None) -> str:
        """Execute research agent - search web for information"""
        
        # Perform web search
        search_results = self.search_web(query)
        
        if not search_results:
            return "No information found from web search."
        
        # Process and summarize search results
        previous_context = ""
        if previous_results and "retrieval" in previous_results:
            previous_context = f"Previous document context:\n{previous_results['retrieval']}"
        
        prompt = f"""You are a research agent. Analyze the search results and provide a comprehensive answer.
Cite sources when possible. Be objective and factual.

Search Results:
{search_results[:2000]}

{previous_context}

Question: {query}

Provide a comprehensive answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant that provides factual information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error in research: {str(e)}"
    
    def search_web(self, query: str) -> str:
        """Search web using DuckDuckGo"""
        try:
            # Using DuckDuckGo Instant Answer API
            url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_html=1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            result_parts = []
            
            if data.get('AbstractText'):
                result_parts.append(f"Summary: {data['AbstractText']}")
            
            if data.get('RelatedTopics'):
                for topic in data['RelatedTopics'][:3]:  # Take first 3
                    if isinstance(topic, dict) and 'Text' in topic:
                        result_parts.append(f"• {topic['Text']}")
            
            return "\n".join(result_parts) if result_parts else "No results found."
        
        except Exception as e:
            return f"Web search error: {str(e)}"