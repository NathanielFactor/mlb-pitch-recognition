"""
Vercel-optimized prediction router for serverless environment.
"""

from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
from app.models.schemas import PredictionResult, StatusResponse, HistoryItem, HistoryResponse
from app.services.vercel_ml_service import vercel_ml_service
from app.core.config import settings

router = APIRouter()

@router.get("/predict/{job_id}", response_model=PredictionResult)
async def get_prediction_result_vercel(job_id: str):
    """Get prediction result for a specific job - Vercel optimized"""
    try:
        result = await vercel_ml_service.get_prediction_result(job_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction: {str(e)}")

@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status_vercel(job_id: str):
    """Get current status and progress of a prediction job - Vercel optimized"""
    try:
        status = vercel_ml_service.get_job_status(job_id)
        return StatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/history", response_model=HistoryResponse)
async def get_prediction_history_vercel():
    """Get user's prediction history from database - Vercel optimized"""
    try:
        history_data = await vercel_ml_service.get_prediction_history()
        
        # Convert database records to HistoryItem objects
        history_items = []
        for record in history_data:
            if record.get("status") == "completed" and record.get("pitch_type"):
                try:
                    # Handle different timestamp formats
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
                    print(f"Error parsing history item: {parse_error}")
                    continue
        
        return HistoryResponse(
            items=history_items,
            total=len(history_items)
        )
        
    except Exception as e:
        # Fallback to empty history if database is unavailable
        print(f"History error: {e}")
        return HistoryResponse(items=[], total=0)

@router.get("/predict/health")
async def predict_health():
    """Health check for prediction service"""
    return {
        "status": "healthy",
        "service": "vercel-predict",
        "environment": "serverless",
        "database_connected": bool(vercel_ml_service)
    }