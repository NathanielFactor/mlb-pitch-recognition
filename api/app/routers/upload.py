from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import UploadResponse
from app.services.file_service import file_service
from app.services.ml_service import ml_service

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for pitch analysis"""
    
    # Validate file
    file_service.validate_file(file)
    
    try:
        # Save uploaded file
        job_id, file_path = await file_service.save_uploaded_file(file)
        
        # Start prediction process
        await ml_service.start_prediction(job_id, file_path, file.filename)
        
        return UploadResponse(
            job_id=job_id,
            message="File uploaded successfully. Processing started.",
            filename=file.filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")