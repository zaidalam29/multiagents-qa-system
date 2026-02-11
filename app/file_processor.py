import PyPDF2
from typing import List, Dict, Any, Tuple
import os
from app.utils.guardrails import validate_content
import tempfile

class FileProcessor:
    @staticmethod
    def process_pdf(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Process PDF file and return chunks with metadata"""
        chunks = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                
                if text:
                    # Validate content
                    if not validate_content(text):
                        continue
                    
                    # Split into chunks
                    page_chunks = FileProcessor._split_text(text, chunk_size, overlap)
                    
                    for chunk_num, chunk in enumerate(page_chunks, 1):
                        chunks.append({
                            "text": chunk,
                            "metadata": {
                                "source": os.path.basename(file_path),
                                "page": page_num,
                                "chunk": chunk_num,
                                "type": "pdf"
                            }
                        })
        
        return chunks
    
    @staticmethod
    def process_text(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Process text file and return chunks"""
        chunks = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            
            if not validate_content(text):
                return []
            
            text_chunks = FileProcessor._split_text(text, chunk_size, overlap)
            
            for chunk_num, chunk in enumerate(text_chunks, 1):
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "chunk": chunk_num,
                        "type": "text"
                    }
                })
        
        return chunks
    
    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                # Try to split at sentence boundary
                split_pos = text.rfind('.', start, end)
                if split_pos > start + chunk_size // 2:
                    end = split_pos + 1
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def process_file(file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Process file based on type"""
        if file_type == "pdf":
            return FileProcessor.process_pdf(file_path)
        elif file_type in ["txt", "md"]:
            return FileProcessor.process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")