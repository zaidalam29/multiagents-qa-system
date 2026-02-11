import hashlib
import hmac
import os
from typing import Optional

# In production, store this in environment variable
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key-change-in-production")

def verify_api_key(api_key: Optional[str]) -> bool:
    """Verify API key"""
    # In production, validate against database or environment
    valid_keys = os.getenv("VALID_API_KEYS", "").split(",")
    
    if not api_key:
        return False
    
    return api_key.strip() in [k.strip() for k in valid_keys if k.strip()]

def verify_file_type(filename: str, allowed_types: list) -> bool:
    """Verify file type"""
    if not filename:
        return False
    
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in allowed_types

def verify_file_size(file_size: int, max_size: int) -> bool:
    """Verify file size"""
    return file_size <= max_size

def generate_signature(data: str) -> str:
    """Generate HMAC signature for data"""
    return hmac.new(
        SECRET_KEY.encode(),
        data.encode(),
        hashlib.sha256
    ).hexdigest()

def verify_signature(data: str, signature: str) -> bool:
    """Verify HMAC signature"""
    expected_signature = generate_signature(data)
    return hmac.compare_digest(expected_signature, signature)