 
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
import tempfile
from typing import List

from app.config import settings
from app.file_processor import FileProcessor
from app.vector_store import VectorStore
from app.utils.guardrails import sanitize_input
from app.utils.security import verify_file_type, verify_file_size

router = APIRouter()
vector_store = VectorStore()

@router.post("/")
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = "documents"
):
    # Validate file type
    if not verify_file_type(file.filename, settings.ALLOWED_FILE_TYPES):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}"
        )
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        
        # Validate file size
        if not verify_file_size(len(content), settings.MAX_FILE_SIZE):
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Process file
        file_type = os.path.splitext(file.filename)[1].lower()
        chunks = FileProcessor.process_file(tmp_path, file_type[1:])  # Remove dot
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid content found in file")
        
        # Prepare for vector store
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Add to vector store
        ids = vector_store.add_documents(
            documents=documents,
            metadatas=metadatas,
            collection_name=collection_name
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "File processed successfully",
                "filename": file.filename,
                "chunks_processed": len(chunks),
                "collection": collection_name,
                "document_ids": ids[:5]  # Return first 5 IDs
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@router.post("/batch")
async def upload_batch(
    files: List[UploadFile] = File(...),
    collection_name: str = "documents"
):
    results = []
    
    for file in files:
        try:
            result = await upload_file(file, collection_name)
            results.append({
                "filename": file.filename,
                "status": "success",
                "details": result.body.decode()
            })
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "details": e.detail
            })
    
    return {"results": results}