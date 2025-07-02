"""
Vercel-optimized ML service for pitch recognition.
This version is designed to work within Vercel's constraints:
- 250MB deployment size limit
- 10-30 second timeout limit
- Stateless serverless functions
"""

import asyncio
import random
import time
import os
from typing import Dict, Any, Optional
from app.models.schemas import PredictionResult
from app.services.database_service import database_service
from app.core.config import settings

class VercelMLService:
    """Vercel-optimized ML service with lazy loading and caching"""
    
    PITCH_TYPES = ["Fastball", "Curveball", "Slider", "Changeup", "Knuckleball", "Cutter"]
    
    def __init__(self):
        self.models_loaded = False
        self.model_cache = {}
        
    async def _load_models_lazy(self):
        """Lazy load models only when needed to reduce cold start time"""
        if self.models_loaded:
            return
            
        try:
            # For now, use lightweight mock models
            # In production, this would load optimized/quantized models
            # from external storage (Supabase Storage, S3, etc.)
            
            if os.getenv('VERCEL'):
                # Running on Vercel - use lightweight inference
                self.model_cache['lightweight'] = True
            else:
                # Running locally - can use full models
                self.model_cache['full'] = True
                
            self.models_loaded = True
            
        except Exception as e:
            print(f"Warning: Failed to load ML models: {e}")
            # Fallback to mock inference
            self.model_cache['mock'] = True
            self.models_loaded = True
    
    async def start_prediction(self, job_id: str, file_path: str, filename: str) -> None:
        """Start async prediction process optimized for Vercel"""
        
        # Create database record if database is available
        if database_service:
            try:
                await database_service.create_prediction(job_id, filename)
            except Exception as e:
                print(f"Database error creating prediction: {e}")
        
        # Start background processing (async to avoid timeout)
        asyncio.create_task(self._process_video_vercel(job_id, file_path, filename))
    
    async def _process_video_vercel(self, job_id: str, file_path: str, filename: str) -> None:
        """Vercel-optimized video processing with timeout handling"""
        start_time = time.time()
        
        try:
            # Load models lazily
            await self._load_models_lazy()
            
            # Quick video validation
            await asyncio.sleep(0.1)  # Simulate quick validation
            
            # Optimized frame extraction (process fewer frames for speed)
            await asyncio.sleep(0.3)  # Simulate faster frame extraction
            
            # Lightweight CNN inference
            await asyncio.sleep(0.5)  # Simulate optimized CNN
            
            # Fast LSTM processing
            await asyncio.sleep(0.3)  # Simulate optimized LSTM
            
            # Generate prediction
            pitch_type = random.choice(self.PITCH_TYPES)
            confidence = random.uniform(0.75, 0.95)
            processing_time = time.time() - start_time
            
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
                except Exception as e:
                    print(f"Database error updating prediction: {e}")
            
            # Clean up temporary files to save space
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
                
        except Exception as e:
            error_message = str(e)
            
            # Update database with error
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
        """Get prediction result from database (Vercel functions are stateless)"""
        # Always try database first since Vercel functions are stateless
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
        
        # Fallback if database unavailable
        return PredictionResult(
            job_id=job_id,
            status="failed",
            error_message="Prediction not found or database unavailable"
        )
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status - for Vercel, always check database"""
        # Vercel functions are stateless, so we can't track progress in memory
        # Return a generic processing status and let frontend poll prediction result
        return {
            "job_id": job_id,
            "status": "processing",
            "progress": 50,  # Generic progress
            "message": "Processing video..."
        }
    
    async def get_prediction_history(self, limit: int = 100) -> list:
        """Get prediction history from database"""
        if database_service:
            try:
                return await database_service.get_prediction_history(limit)
            except Exception as e:
                print(f"Database error getting history: {e}")
        
        return []

# Global instance for Vercel
vercel_ml_service = VercelMLService()