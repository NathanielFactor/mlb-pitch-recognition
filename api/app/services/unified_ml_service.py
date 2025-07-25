"""
Unified ML service for pitch recognition that adapts to both local development
and Vercel deployment environments automatically.
"""

import asyncio
import random
import time
import os
from typing import Dict, Any, Optional
from app.models.schemas import PredictionResult
from app.services.database_service import database_service
from app.core.config import settings


class UnifiedMLService:
    """Unified ML service that adapts to environment constraints"""
    
    PITCH_TYPES = ["Fastball", "Curveball", "Slider", "Changeup", "Knuckleball", "Cutter"]
    
    def __init__(self):
        self.processing_jobs: Dict[str, Dict[str, Any]] = {}
        self.models_loaded = False
        self.model_cache = {}
        self.is_vercel = os.getenv('VERCEL') is not None
        
    @property
    def supports_in_memory_tracking(self) -> bool:
        """Check if current environment supports in-memory job tracking"""
        return not self.is_vercel
    
    async def _load_models_lazy(self):
        """Lazy load models only when needed to reduce cold start time"""
        if self.models_loaded:
            return
            
        try:
            # For now, use lightweight mock models
            # In production, this would load optimized/quantized models
            # from external storage (Supabase Storage, S3, etc.)
            
            if self.is_vercel:
                # Running on Vercel - use lightweight inference
                self.model_cache['lightweight'] = True
                if not self.is_vercel:  # Only print in dev environment
                    print("Loaded lightweight models for Vercel environment")
            else:
                # Running locally - can use full models
                self.model_cache['full'] = True
                print("Loaded full models for local development")
                
            self.models_loaded = True
            
        except Exception as e:
            print(f"Warning: Failed to load ML models: {e}")
            # Fallback to mock inference
            self.model_cache['mock'] = True
            self.models_loaded = True
    
    async def start_prediction(self, job_id: str, file_path: str, filename: str) -> None:
        """Start async prediction process with environment-specific handling"""
        
        # For Vercel, validate database connection before starting
        if self.is_vercel and not database_service:
            raise Exception("Database service unavailable - cannot start prediction")
        
        # Store initial state in memory for progress tracking (dev environment only)
        if self.supports_in_memory_tracking:
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
                if not self.is_vercel:  # Only print in dev environment
                    print(f"Database record created for job: {job_id}")
            except Exception as e:
                error_msg = f"Database error creating prediction: {e}"
                if not self.is_vercel:
                    print(error_msg)
                if self.is_vercel:
                    raise Exception(error_msg)
        
        # Start background processing
        asyncio.create_task(self._process_video(job_id, file_path, filename))
    
    async def _process_video(self, job_id: str, file_path: str, filename: str) -> None:
        """Process video with environment-specific optimizations"""
        start_time = time.time()
        
        try:
            if not self.is_vercel:
                print(f"Starting video processing for job: {job_id}")
            
            # Load models lazily
            await self._load_models_lazy()
            
            if self.is_vercel:
                # Vercel-optimized processing (faster, fewer steps)
                await self._process_video_vercel_optimized(job_id)
            else:
                # Local development processing (detailed progress tracking)
                await self._process_video_detailed(job_id)
            
            # Generate prediction
            pitch_type = random.choice(self.PITCH_TYPES)
            confidence = random.uniform(0.7, 0.95)
            processing_time = time.time() - start_time
            
            if not self.is_vercel:
                print(f"Prediction generated for job {job_id}: {pitch_type} ({confidence:.2f})")
            
            # Update in-memory status (dev environment only)
            if self.supports_in_memory_tracking:
                self.processing_jobs[job_id].update({
                    "status": "completed",
                    "progress": 100,
                    "pitch_type": pitch_type,
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "message": "Prediction completed"
                })
            
            # Update database with results
            if database_service:
                try:
                    await database_service.update_prediction_status(
                        job_id=job_id,
                        status="completed",
                        pitch_type=pitch_type,
                        confidence=confidence,
                        processing_time=processing_time
                    )
                    if not self.is_vercel:
                        print(f"Database updated successfully for job: {job_id}")
                except Exception as e:
                    error_msg = f"Database error updating prediction: {e}"
                    if not self.is_vercel:
                        print(error_msg)
                    if self.is_vercel:
                        raise Exception(error_msg)
            
            # Clean up temporary files (especially important on Vercel)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    if not self.is_vercel:
                        print(f"Temporary file cleaned up for job: {job_id}")
            except Exception as cleanup_error:
                if not self.is_vercel:
                    print(f"Warning: Failed to cleanup file for job {job_id}: {cleanup_error}")
                
        except Exception as e:
            error_message = str(e)
            if not self.is_vercel:
                print(f"Processing failed for job {job_id}: {error_message}")
            
            # Update in-memory status (dev environment only)
            if self.supports_in_memory_tracking:
                self.processing_jobs[job_id].update({
                    "status": "failed",
                    "error_message": error_message,
                    "message": "Processing failed"
                })
            
            # Update database with error
            if database_service:
                try:
                    await database_service.update_prediction_status(
                        job_id=job_id,
                        status="failed",
                        error_message=error_message
                    )
                    if not self.is_vercel:
                        print(f"Database updated with error for job: {job_id}")
                except Exception as db_error:
                    if not self.is_vercel:
                        print(f"Database error updating failed prediction: {db_error}")
    
    async def _process_video_detailed(self, job_id: str) -> None:
        """Detailed processing with progress updates for local development"""
        # Simulate video loading and preprocessing
        await asyncio.sleep(0.5)
        if self.supports_in_memory_tracking:
            self.processing_jobs[job_id]["progress"] = 20
            self.processing_jobs[job_id]["message"] = "Loading video..."
        
        # Simulate frame extraction
        await asyncio.sleep(1.0)
        if self.supports_in_memory_tracking:
            self.processing_jobs[job_id]["progress"] = 40
            self.processing_jobs[job_id]["message"] = "Extracting frames..."
        
        # Simulate CNN feature extraction
        await asyncio.sleep(1.5)
        if self.supports_in_memory_tracking:
            self.processing_jobs[job_id]["progress"] = 70
            self.processing_jobs[job_id]["message"] = "Analyzing pitch mechanics..."
        
        # Simulate LSTM processing
        await asyncio.sleep(1.0)
        if self.supports_in_memory_tracking:
            self.processing_jobs[job_id]["progress"] = 90
            self.processing_jobs[job_id]["message"] = "Classifying pitch type..."
        
        # Final step
        await asyncio.sleep(0.5)
    
    async def _process_video_vercel_optimized(self, job_id: str) -> None:
        """Optimized processing for Vercel environment"""
        # Quick video validation
        await asyncio.sleep(0.1)
        
        # Optimized frame extraction (process fewer frames for speed)
        await asyncio.sleep(0.3)
        
        # Lightweight CNN inference  
        await asyncio.sleep(0.5)
        
        # Fast LSTM processing
        await asyncio.sleep(0.3)
    
    async def get_prediction_result(self, job_id: str) -> PredictionResult:
        """Get prediction result, adapting to environment capabilities"""
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
                if not self.is_vercel:
                    print(f"Database error getting prediction: {e}")
        
        # Fallback to memory if database unavailable (dev environment only)
        if self.supports_in_memory_tracking and job_id in self.processing_jobs:
            job = self.processing_jobs[job_id]
            return PredictionResult(
                job_id=job_id,
                status=job["status"],
                pitch_type=job.get("pitch_type"),
                confidence=job.get("confidence"),
                processing_time=job.get("processing_time"),
                error_message=job.get("error_message")
            )
        
        # Final fallback
        error_msg = "Job not found" if not self.is_vercel else "Prediction not found or database unavailable"
        return PredictionResult(
            job_id=job_id,
            status="failed",
            error_message=error_msg
        )
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get current job status with environment-appropriate response"""
        # In development, use in-memory tracking for real-time progress
        if self.supports_in_memory_tracking and job_id in self.processing_jobs:
            job = self.processing_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job["status"],
                "progress": job.get("progress", 0),
                "message": job.get("message", "")
            }
        
        # In Vercel, return generic status since functions are stateless
        if self.is_vercel:
            return {
                "job_id": job_id,
                "status": "processing",
                "progress": 50,  # Generic progress
                "message": "Processing video..."
            }
        
        # Fallback for unknown jobs
        return {
            "job_id": job_id,
            "status": "failed",
            "message": "Job not found"
        }
    
    async def get_prediction_history(self, limit: int = 100) -> list:
        """Get prediction history from database"""
        if database_service:
            try:
                return await database_service.get_prediction_history(limit)
            except Exception as e:
                if not self.is_vercel:
                    print(f"Database error getting history: {e}")
        
        # Fallback to empty list if database unavailable
        return []


# Global instance - works in both environments
ml_service = UnifiedMLService()