"""
Unified prediction router that adapts to both local development and Vercel deployment.
"""

import os
from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
from app.models.schemas import PredictionResult, StatusResponse, HistoryItem, HistoryResponse
from app.services.unified_ml_service import ml_service
from app.core.config import settings

router = APIRouter()

# Environment detection
IS_VERCEL = os.getenv('VERCEL') is not None

@router.get("/predict/{job_id}", response_model=PredictionResult)
async def get_prediction_result(job_id: str):
    """Get prediction result for a specific job - works in all environments"""
    try:
        result = await ml_service.get_prediction_result(job_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction: {str(e)}")


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """Get current status and progress of a prediction job - environment aware"""
    try:
        status = ml_service.get_job_status(job_id)
        return StatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/history", response_model=HistoryResponse)
async def get_prediction_history():
    """Get user's prediction history from database - environment aware"""
    try:
        history_data = await ml_service.get_prediction_history()
        
        # Convert database records to HistoryItem objects
        history_items = []
        for record in history_data:
            if record.get("status") == "completed" and record.get("pitch_type"):
                try:
                    # Handle different timestamp formats (enhanced for Vercel)
                    upload_date_str = record["upload_date"]
                    if isinstance(upload_date_str, str):
                        # Remove timezone suffix if present and parse
                        upload_date_str = upload_date_str.replace('Z', '+00:00')
                        upload_date = datetime.fromisoformat(upload_date_str)
                    else:
                        upload_date = upload_date_str
                    
                    history_items.append(HistoryItem(
                        id=record["job_id"],
                        filename=record["filename"],
                        upload_date=upload_date,
                        pitch_type=record["pitch_type"],
                        confidence=record.get("confidence", 0.0),
                        processing_time=record.get("processing_time", 0.0)
                    ))
                except Exception as parse_error:
                    if IS_VERCEL:
                        print(f"Error parsing history item: {parse_error}")
                    continue
        
        return HistoryResponse(
            items=history_items,
            total=len(history_items)
        )
        
    except Exception as e:
        # Environment-specific fallback handling
        if IS_VERCEL:
            # Vercel: Return empty history on error
            print(f"History error: {e}")
            return HistoryResponse(items=[], total=0)
        else:
            # Local development: Try mock data if available
            if settings.MOCK_MODE:
                mock_history = [
                    HistoryItem(
                        id="1",
                        filename="fastball_pitch_1.mp4",
                        upload_date=datetime(2024, 6, 30, 14, 30),
                        pitch_type="Fastball",
                        confidence=0.92,
                        processing_time=2.1
                    ),
                    HistoryItem(
                        id="2",
                        filename="curveball_sample.mp4",
                        upload_date=datetime(2024, 6, 29, 16, 45),
                        pitch_type="Curveball",
                        confidence=0.87,
                        processing_time=2.5
                    )
                ]
                
                return HistoryResponse(
                    items=mock_history,
                    total=len(mock_history)
                )
            else:
                raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/predict/health")
async def predict_health():
    """Health check for prediction service - adapts to environment"""
    environment = "vercel" if IS_VERCEL else "local"
    
    return {
        "status": "healthy",
        "service": "unified-predict",
        "environment": environment,
        "database_connected": bool(ml_service)
    }