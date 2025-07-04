"""
Vercel-optimized upload router with size and timeout constraints handling.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import UploadResponse
from app.services.file_service import file_service
from app.services.vercel_ml_service import vercel_ml_service
import os

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_video_vercel(file: UploadFile = File(...)):
    """Upload a video file for pitch analysis - Vercel optimized"""
    
    # Enhanced validation for Vercel constraints
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file size more strictly for Vercel
    if hasattr(file, 'size') and file.size:
        # Limit to 50MB for Vercel to ensure processing within timeout
        max_size = 50 * 1024 * 1024  # 50MB
        if file.size > max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large for serverless processing. Maximum size: 50MB"
            )
    
    # Validate file type
    try:
        file_service.validate_file(file)
    except HTTPException as e:
        print(f"File validation error: {e.detail}")
        raise
    
    # Check database connection before processing
    from app.services.database_service import database_service
    if not database_service:
        raise HTTPException(
            status_code=503, 
            detail="Database unavailable. Please check Supabase configuration."
        )
    
    try:
        # Save uploaded file (this will be in /tmp on Vercel)
        job_id, file_path = await file_service.save_uploaded_file(file)
        print(f"File saved successfully: {job_id}, {file_path}")
        
        # Start prediction process with Vercel-optimized service
        await vercel_ml_service.start_prediction(job_id, file_path, file.filename)
        print(f"Prediction started for job: {job_id}")
        
        return UploadResponse(
            job_id=job_id,
            message="File uploaded successfully. Processing started.",
            filename=file.filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        print(f"Upload error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/upload/health")
async def upload_health():
    """Health check for upload service"""
    return {
        "status": "healthy",
        "service": "vercel-upload",
        "max_file_size": "50MB",
        "supported_formats": [".mp4", ".avi", ".mov"]
    }