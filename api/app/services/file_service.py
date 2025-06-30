import os
import aiofiles
from fastapi import UploadFile, HTTPException
from typing import Tuple
import uuid
from app.core.config import settings

class FileService:
    """Service for handling file uploads and validation"""
    
    @staticmethod
    def validate_file(file: UploadFile) -> None:
        """Validate uploaded file"""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_ext} not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Check file size (this is approximate since we get it from headers)
        if hasattr(file, 'size') and file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
    
    @staticmethod
    async def save_uploaded_file(file: UploadFile) -> Tuple[str, str]:
        """Save uploaded file and return (job_id, file_path)"""
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create unique filename
        file_ext = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{job_id}{file_ext}"
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
        
        try:
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            return job_id, file_path
            
        except Exception as e:
            # Clean up partial file if something went wrong
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    @staticmethod
    def cleanup_file(file_path: str) -> None:
        """Remove file from filesystem"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            # Log error but don't raise - cleanup is best effort
            pass

file_service = FileService()