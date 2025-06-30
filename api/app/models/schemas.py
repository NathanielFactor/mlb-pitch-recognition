from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime

class UploadResponse(BaseModel):
    job_id: str
    message: str
    filename: str

class PredictionResult(BaseModel):
    job_id: str
    status: Literal["processing", "completed", "failed"]
    pitch_type: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

class StatusResponse(BaseModel):
    job_id: str
    status: Literal["processing", "completed", "failed"]
    progress: Optional[int] = None  # 0-100
    message: Optional[str] = None

class HistoryItem(BaseModel):
    id: str
    filename: str
    upload_date: datetime
    pitch_type: str
    confidence: float
    processing_time: float

class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int

class HealthResponse(BaseModel):
    status: str
    mock_mode: bool
    timestamp: datetime