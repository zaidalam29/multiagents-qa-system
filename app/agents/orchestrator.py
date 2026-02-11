from typing import Dict, Any, List, Optional
from app.config import settings

class OrchestratorAgent:
    def __init__(self, retrieval_agent, research_agent, validation_agent):
        self.client, self.model, self.provider = settings.get_llm_client()
        self.agents = {
            "retrieval": retrieval_agent,
            "research": research_agent,
            "validation": validation_agent
        }
    
    def decide_agents(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """Decide which agents to use for the query"""
        system_prompt = """You are an intelligent orchestrator. Based on the user query, decide which agent(s) to use:
        1. Use RETRIEVAL agent when query is about uploaded documents
        2. Use RESEARCH agent when query requires external information
        3. Use VALIDATION agent to verify and format the final answer
        
        Available agents: retrieval, research, validation
        
        Provide your decision as a JSON array like: ["retrieval", "validation"]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\nContext: {context}"}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result.get("agents", ["retrieval", "validation"])
        
        except Exception as e:
            print(f"Error in orchestrator: {e}")
            return ["retrieval", "validation"]
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimized query processing"""
        
        # Skip validation if retrieval has good answer
        agents_to_use = ["retrieval"]  # Always start with retrieval
        
        # Only add research if explicitly requested
        if context and context.get('use_internet'):
            agents_to_use.append("research")
        
        # Only add validation if we have multiple sources
        if len(agents_to_use) > 1:
            agents_to_use.append("validation")
        
        results = {}
        
        for agent_name in agents_to_use:
            if agent_name in self.agents:
                try:
                    agent = self.agents[agent_name]
                    
                    # Skip if retrieval returned "no information"
                    if agent_name == "research" and "retrieval" in results:
                        retrieval_answer = results["retrieval"].lower()
                        if "no relevant" in retrieval_answer or "not found" in retrieval_answer:
                            # Actually, we should do research when retrieval fails
                            pass
                    
                    result = agent.execute(query, context, previous_results=results)
                    results[agent_name] = result
                    
                except Exception as e:
                    results[agent_name] = f"Error: {str(e)}"
        
        # Combine results intelligently
        if "validation" in results:
            final_answer = results["validation"]
        elif "retrieval" in results:
            final_answer = results["retrieval"]
        elif "research" in results:
            final_answer = results["research"]
        else:
            final_answer = "Unable to generate answer."
        
        return {
            "answer": final_answer,
            "agents_used": agents_to_use,
            "intermediate_results": results
        }
    
    def _combine_results(self, results: Dict[str, Any]) -> str:
        """Combine results from multiple agents"""
        if "validation" in results:
            return results["validation"]
        elif "retrieval" in results:
            return results["retrieval"]
        elif "research" in results:
            return results["research"]
        else:
            return "I couldn't find a suitable answer."