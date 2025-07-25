"""
Unified upload router that adapts to both local development and Vercel deployment.
"""

import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import UploadResponse
from app.services.file_service import file_service
from app.services.unified_ml_service import ml_service

router = APIRouter()

# Environment detection
IS_VERCEL = os.getenv('VERCEL') is not None

@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for pitch analysis - works in all environments"""
    
    # Enhanced validation for filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Environment-specific file size validation
    if hasattr(file, 'size') and file.size:
        if IS_VERCEL:
            # Stricter limit for Vercel to ensure processing within timeout
            max_size = 50 * 1024 * 1024  # 50MB
            if file.size > max_size:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File too large for serverless processing. Maximum size: 50MB"
                )
        else:
            # More generous limit for local development
            max_size = 100 * 1024 * 1024  # 100MB
            if file.size > max_size:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File too large. Maximum size: 100MB"
                )
    
    # Validate file type
    try:
        file_service.validate_file(file)
    except HTTPException as e:
        if IS_VERCEL:
            print(f"File validation error: {e.detail}")
        raise
    
    # Check database connection for Vercel environment
    if IS_VERCEL:
        from app.services.database_service import database_service
        if not database_service:
            raise HTTPException(
                status_code=503, 
                detail="Database unavailable. Please check Supabase configuration."
            )
    
    try:
        # Save uploaded file
        job_id, file_path = await file_service.save_uploaded_file(file)
        if IS_VERCEL:
            print(f"File saved successfully: {job_id}, {file_path}")
        
        # Start prediction process with unified service
        await ml_service.start_prediction(job_id, file_path, file.filename)
        if IS_VERCEL:
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
        if IS_VERCEL:
            print(f"Upload error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/upload/health")
async def upload_health():
    """Health check for upload service - adapts to environment"""
    max_file_size = "50MB" if IS_VERCEL else "100MB"
    environment = "vercel" if IS_VERCEL else "local"
    
    return {
        "status": "healthy",
        "service": "unified-upload",
        "environment": environment,
        "max_file_size": max_file_size,
        "supported_formats": [".mp4", ".avi", ".mov"]
    }