from typing import Dict, Any, Optional
from app.config import settings
from app.utils.guardrails import validate_response

class ValidationAgent:
    def __init__(self):
        self.client, self.model, self.provider = settings.get_llm_client()
    
    def execute(self, query: str, context: Optional[Dict] = None, 
                previous_results: Optional[Dict] = None) -> str:
        """Execute validation agent"""
        
        if not previous_results:
            return "No results to validate."
        
        # Combine all previous results
        combined_results = self._combine_previous_results(previous_results)
        
        # Validate and format final answer
        prompt = f"""You are a validation and formatting agent. Your tasks:
1. Verify the accuracy and consistency of information
2. Remove any contradictions
3. Format the answer professionally
4. Add citations if available
5. Ensure the answer directly addresses the query

Available information from different agents:
{combined_results}

Original query: {query}

Please provide the final validated answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You validate and format answers professionally."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=600
            )
            
            final_answer = response.choices[0].message.content
            
            # Apply guardrails
            if not validate_response(final_answer):
                final_answer = "I cannot provide an answer that might violate content policies."
            
            return final_answer
        
        except Exception as e:
            return f"Validation error: {str(e)}"
    
    def _combine_previous_results(self, previous_results: Dict[str, Any]) -> str:
        """Combine results from different agents"""
        combined = []
        
        for agent_name, result in previous_results.items():
            combined.append(f"--- From {agent_name.upper()} agent ---\n{result}")
        
        return "\n\n".join(combined)