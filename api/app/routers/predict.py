from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
from app.models.schemas import PredictionResult, StatusResponse, HistoryItem, HistoryResponse
from app.services.ml_service import ml_service
from app.core.config import settings

router = APIRouter()

@router.get("/predict/{job_id}", response_model=PredictionResult)
async def get_prediction_result(job_id: str):
    """Get prediction result for a specific job"""
    try:
        result = await ml_service.get_prediction_result(job_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction: {str(e)}")

@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """Get current status and progress of a prediction job"""
    try:
        status = ml_service.get_job_status(job_id)
        return StatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/history", response_model=HistoryResponse)
async def get_prediction_history():
    """Get user's prediction history from database"""
    try:
        history_data = await ml_service.get_prediction_history()
        
        # Convert database records to HistoryItem objects
        history_items = []
        for record in history_data:
            if record.get("status") == "completed" and record.get("pitch_type"):
                history_items.append(HistoryItem(
                    id=record["job_id"],
                    filename=record["filename"],
                    upload_date=datetime.fromisoformat(record["upload_date"].replace('Z', '+00:00')),
                    pitch_type=record["pitch_type"],
                    confidence=record.get("confidence", 0.0),
                    processing_time=record.get("processing_time", 0.0)
                ))
        
        return HistoryResponse(
            items=history_items,
            total=len(history_items)
        )
        
    except Exception as e:
        # Fallback to mock data if database is unavailable
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