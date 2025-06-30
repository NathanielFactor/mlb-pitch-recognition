from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
from app.models.schemas import PredictionResult, StatusResponse, HistoryItem, HistoryResponse
from app.services.mock_ml_service import mock_ml_service

router = APIRouter()

@router.get("/predict/{job_id}", response_model=PredictionResult)
async def get_prediction_result(job_id: str):
    """Get prediction result for a specific job"""
    try:
        result = mock_ml_service.get_prediction_result(job_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction: {str(e)}")

@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """Get current status and progress of a prediction job"""
    try:
        status = mock_ml_service.get_job_status(job_id)
        return StatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/history", response_model=HistoryResponse)
async def get_prediction_history():
    """Get user's prediction history (mock data for now)"""
    # Mock history data
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
        ),
        HistoryItem(
            id="3",
            filename="slider_demo.mp4",
            upload_date=datetime(2024, 6, 28, 10, 15),
            pitch_type="Slider",
            confidence=0.79,
            processing_time=1.9
        ),
        HistoryItem(
            id="4",
            filename="changeup_example.mov",
            upload_date=datetime(2024, 6, 27, 13, 20),
            pitch_type="Changeup",
            confidence=0.85,
            processing_time=2.3
        )
    ]
    
    return HistoryResponse(
        items=mock_history,
        total=len(mock_history)
    )