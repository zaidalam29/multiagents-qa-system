import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
import openai
import numpy as np
import httpx

class Settings(BaseSettings):
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API Key")
    OPENAI_MODEL: str = Field("gpt-4-turbo-preview", description="OpenAI Model")
    EMBEDDING_MODEL: str = Field("text-embedding-3-small", description="Embedding Model")
    
    # OpenRouter Configuration
    OPENROUTER_API_KEY: Optional[str] = Field(None, description="OpenRouter API Key")
    OPENROUTER_BASE_URL: str = Field("https://openrouter.ai/api/v1", description="OpenRouter Base URL")
    OPENROUTER_MODEL: str = Field("anthropic/claude-3-opus", description="OpenRouter Model")
    
    # Vector Database
    VECTOR_DB_PATH: str = Field("./vector_store", description="Vector Database Path")
    
    # Performance settings
    USE_LOCAL_EMBEDDINGS: bool = Field(False, description="Use local embeddings for faster search")
    REQUEST_TIMEOUT: int = Field(30, description="Request timeout in seconds")
    
    # Application
    APP_HOST: str = Field("0.0.0.0", description="Application Host")
    APP_PORT: int = Field(8000, description="Application Port")
    DEBUG: bool = Field(False, description="Debug Mode")
    
    # Security
    API_KEY_HEADER: str = Field("X-API-Key", description="API Key Header")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def ALLOWED_FILE_TYPES(self) -> List[str]:
        """Allowed file types"""
        return [".pdf", ".txt", ".md"]
    
    @property
    def MAX_FILE_SIZE(self) -> int:
        """Maximum file size (10MB)"""
        return 10 * 1024 * 1024
    
    def get_llm_client(self):
        """Get LLM client with fallback and timeout"""
        timeout = httpx.Timeout(self.REQUEST_TIMEOUT, connect=5.0)
        
        if self.OPENAI_API_KEY:
            return openai.OpenAI(
                api_key=self.OPENAI_API_KEY,
                timeout=timeout
            ), self.OPENAI_MODEL, "openai"
        elif self.OPENROUTER_API_KEY:
            return openai.OpenAI(
                base_url=self.OPENROUTER_BASE_URL,
                api_key=self.OPENROUTER_API_KEY,
                timeout=timeout
            ), self.OPENROUTER_MODEL, "openrouter"
        else:
            raise ValueError("No LLM API key configured")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings - optimized for performance"""
        if not texts:
            return []
        
        if self.USE_LOCAL_EMBEDDINGS:
            return self._get_local_embeddings(texts)
        else:
            return self._get_openai_embeddings(texts)
    
    def _get_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fast local embeddings"""
        try:
            # Lazy import to avoid dependency if not used
            from sentence_transformers import SentenceTransformer
            
            if not hasattr(self, '_local_model'):
                print("Loading local embedding model...")
                self._local_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Batch encode
            embeddings = self._local_model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
            
        except ImportError:
            print("Warning: sentence-transformers not installed, using simple embeddings")
            return self._get_simple_embeddings(texts)
        except Exception as e:
            print(f"Local embedding error: {e}")
            return self._get_simple_embeddings(texts)
    
    def _get_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """OpenAI embeddings with batching"""
        try:
            client, _, _ = self.get_llm_client()
            
            # Single batch call for better performance
            response = client.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=texts
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            print(f"OpenAI embedding error: {e}")
            return self._get_simple_embeddings(texts)
    
    def _get_simple_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Simple fallback embeddings"""
        embeddings = []
        for text in texts:
            embedding = [0.0] * 384
            text_lower = text.lower()
            
            for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
                if i < 26:
                    embedding[i] = text_lower.count(char) / max(len(text), 1)
            
            embedding[26] = len(text) / 1000.0
            embedding[27] = text.count('.') / 10.0
            embedding[28] = text.count(' ') / 100.0
            
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            for i in range(31, 384):
                hash_part = int(text_hash[i % 32], 16) / 16.0
                embedding[i] = hash_part
            
            embeddings.append(embedding)
        
        return embeddings

settings = Settings()