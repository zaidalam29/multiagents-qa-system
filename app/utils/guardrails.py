import re
from typing import List, Tuple
import numpy as np

class Guardrails:
    # Define sensitive patterns (can be expanded)
    SENSITIVE_PATTERNS = [
        r'\b(credit\s*card|ssn|social\s*security|password|secret)\b',
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
        r'\b\d{16}\b',  # Credit card numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    ]
    
    TOXIC_PATTERNS = [
        r'\b(kill|murder|harm|hurt|attack)\s+(you|them|us)\b',
        r'\b(hate|despise|loathe)\s+(you|them|gays|jews|muslims)\b',
        # Add more patterns as needed
    ]
    
    @staticmethod
    def validate_content(text: str) -> bool:
        """Validate content for sensitive information"""
        if not text or len(text.strip()) < 10:
            return False
        
        # Check for sensitive patterns
        for pattern in Guardrails.SENSITIVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # Check for toxic content
        for pattern in Guardrails.TOXIC_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def validate_response(response: str) -> bool:
        """Validate AI response"""
        if not response:
            return False
        
        # Check length
        if len(response) > 5000:  # Too long
            return False
        
        # Check for refusal patterns
        refusal_patterns = [
            r"i cannot", r"i can't", r"as an ai", r"i'm sorry",
            r"i don't know how", r"not appropriate", r"not ethical"
        ]
        
        refusal_count = sum(1 for pattern in refusal_patterns 
                          if re.search(pattern, response, re.IGNORECASE))
        
        if refusal_count > 2:
            return False
        
        return True
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input"""
        # Remove potentially harmful characters
        text = re.sub(r'[<>\"\']', '', text)
        
        # Limit length
        if len(text) > 1000:
            text = text[:1000]
        
        return text.strip()

def validate_content(text: str) -> bool:
    return Guardrails.validate_content(text)

def validate_response(response: str) -> bool:
    return Guardrails.validate_response(response)

def sanitize_input(text: str) -> str:
    return Guardrails.sanitize_input(text)