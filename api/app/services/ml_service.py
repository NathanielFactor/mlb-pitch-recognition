import asyncio
import random
import time
from typing import Dict, Any, Optional
from app.models.schemas import PredictionResult
from app.services.database_service import database_service
from app.core.config import settings

class MLService:
    """ML service that integrates with database for persistence"""
    
    PITCH_TYPES = ["Fastball", "Curveball", "Slider", "Changeup", "Knuckleball", "Cutter"]
    
    def __init__(self):
        self.processing_jobs: Dict[str, Dict[str, Any]] = {}
    
    async def start_prediction(self, job_id: str, file_path: str, filename: str) -> None:
        """Start async prediction process and store in database"""
        # Store initial state in memory for progress tracking
        self.processing_jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "file_path": file_path,
            "filename": filename
        }
        
        # Create database record if database is available
        if database_service:
            try:
                await database_service.create_prediction(job_id, filename)
            except Exception as e:
                print(f"Database error creating prediction: {e}")
        
        # Start background processing
        asyncio.create_task(self._process_video(job_id))
    
    async def _process_video(self, job_id: str) -> None:
        """Process video with realistic delays and update database"""
        start_time = time.time()
        
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
            processing_time = time.time() - start_time
            
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
            
            # Update database with results if available
            if database_service:
                try:
                    await database_service.update_prediction_status(
                        job_id=job_id,
                        status="completed",
                        pitch_type=pitch_type,
                        confidence=confidence,
                        processing_time=processing_time
                    )
                except Exception as e:
                    print(f"Database error updating prediction: {e}")
            
        except Exception as e:
            error_message = str(e)
            self.processing_jobs[job_id].update({
                "status": "failed",
                "error_message": error_message,
                "message": "Processing failed"
            })
            
            # Update database with error if available
            if database_service:
                try:
                    await database_service.update_prediction_status(
                        job_id=job_id,
                        status="failed",
                        error_message=error_message
                    )
                except Exception as db_error:
                    print(f"Database error updating failed prediction: {db_error}")
    
    async def get_prediction_result(self, job_id: str) -> PredictionResult:
        """Get prediction result, preferring database over memory"""
        # Try to get from database first
        if database_service:
            try:
                db_result = await database_service.get_prediction(job_id)
                if db_result:
                    return PredictionResult(
                        job_id=job_id,
                        status=db_result["status"],
                        pitch_type=db_result.get("pitch_type"),
                        confidence=db_result.get("confidence"),
                        processing_time=db_result.get("processing_time"),
                        error_message=db_result.get("error_message")
                    )
            except Exception as e:
                print(f"Database error getting prediction: {e}")
        
        # Fallback to memory if database unavailable or job not found
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
        """Get current job status with progress (from memory for real-time updates)"""
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
    
    async def get_prediction_history(self, limit: int = 100) -> list:
        """Get prediction history from database"""
        if database_service:
            try:
                return await database_service.get_prediction_history(limit)
            except Exception as e:
                print(f"Database error getting history: {e}")
        
        # Fallback to empty list if database unavailable
        return []

# Global instance
ml_service = MLService()