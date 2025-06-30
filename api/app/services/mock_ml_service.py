import asyncio
import random
from typing import Dict, Any
from app.models.schemas import PredictionResult

class MockMLService:
    """Mock ML service that simulates pitch prediction"""
    
    PITCH_TYPES = ["Fastball", "Curveball", "Slider", "Changeup", "Knuckleball", "Cutter"]
    
    def __init__(self):
        self.processing_jobs: Dict[str, Dict[str, Any]] = {}
    
    async def start_prediction(self, job_id: str, file_path: str) -> None:
        """Start async prediction process"""
        self.processing_jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "file_path": file_path
        }
        
        # Start background processing
        asyncio.create_task(self._process_video(job_id))
    
    async def _process_video(self, job_id: str) -> None:
        """Simulate video processing with realistic delays"""
        try:
            # Simulate video loading and preprocessing
            await asyncio.sleep(0.5)
            self.processing_jobs[job_id]["progress"] = 20
            self.processing_jobs[job_id]["message"] = "Loading video..."
            
            # Simulate frame extraction
            await asyncio.sleep(1.0)
            self.processing_jobs[job_id]["progress"] = 40
            self.processing_jobs[job_id]["message"] = "Extracting frames..."
            
            # Simulate CNN feature extraction
            await asyncio.sleep(1.5)
            self.processing_jobs[job_id]["progress"] = 70
            self.processing_jobs[job_id]["message"] = "Analyzing pitch mechanics..."
            
            # Simulate LSTM processing
            await asyncio.sleep(1.0)
            self.processing_jobs[job_id]["progress"] = 90
            self.processing_jobs[job_id]["message"] = "Classifying pitch type..."
            
            # Generate realistic prediction
            pitch_type = random.choice(self.PITCH_TYPES)
            confidence = random.uniform(0.7, 0.95)
            processing_time = random.uniform(2.0, 4.0)
            
            # Complete processing
            await asyncio.sleep(0.5)
            self.processing_jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "pitch_type": pitch_type,
                "confidence": confidence,
                "processing_time": processing_time,
                "message": "Prediction completed"
            })
            
        except Exception as e:
            self.processing_jobs[job_id].update({
                "status": "failed",
                "error_message": str(e),
                "message": "Processing failed"
            })
    
    def get_prediction_result(self, job_id: str) -> PredictionResult:
        """Get prediction result for a job"""
        if job_id not in self.processing_jobs:
            return PredictionResult(
                job_id=job_id,
                status="failed",
                error_message="Job not found"
            )
        
        job = self.processing_jobs[job_id]
        
        return PredictionResult(
            job_id=job_id,
            status=job["status"],
            pitch_type=job.get("pitch_type"),
            confidence=job.get("confidence"),
            processing_time=job.get("processing_time"),
            error_message=job.get("error_message")
        )
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get current job status with progress"""
        if job_id not in self.processing_jobs:
            return {
                "job_id": job_id,
                "status": "failed",
                "message": "Job not found"
            }
        
        job = self.processing_jobs[job_id]
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job.get("progress", 0),
            "message": job.get("message", "")
        }

# Global instance
mock_ml_service = MockMLService()