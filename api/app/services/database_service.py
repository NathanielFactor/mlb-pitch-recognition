from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from supabase import create_client, Client
from ..core.config import settings

class DatabaseService:
    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("Supabase URL and KEY are required")
        
        try:
            self.supabase: Client = create_client(
                settings.SUPABASE_URL, 
                settings.SUPABASE_KEY
            )
            self.service_client: Client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_KEY or settings.SUPABASE_KEY
            )
        except Exception as e:
            print(f"Failed to initialize Supabase client: {e}")
            raise
    
    async def create_prediction(self, job_id: str, filename: str) -> Dict[str, Any]:
        """Create a new prediction record in the database."""
        try:
            result = self.supabase.table('predictions').insert({
                'job_id': job_id,
                'filename': filename,
                'status': 'processing',
                'upload_date': datetime.utcnow().isoformat()
            }).execute()
            
            if result.data:
                return result.data[0]
            else:
                raise Exception("Failed to create prediction record")
        except Exception as e:
            raise Exception(f"Database error creating prediction: {str(e)}")
    
    async def update_prediction_status(self, job_id: str, status: str, 
                                     pitch_type: Optional[str] = None,
                                     confidence: Optional[float] = None,
                                     processing_time: Optional[float] = None,
                                     error_message: Optional[str] = None) -> Dict[str, Any]:
        """Update prediction status and results."""
        try:
            update_data = {
                'status': status,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            if pitch_type is not None:
                update_data['pitch_type'] = pitch_type
            if confidence is not None:
                update_data['confidence'] = confidence
            if processing_time is not None:
                update_data['processing_time'] = processing_time
            if error_message is not None:
                update_data['error_message'] = error_message
            
            result = self.supabase.table('predictions').update(update_data).eq('job_id', job_id).execute()
            
            if result.data:
                return result.data[0]
            else:
                raise Exception("Failed to update prediction record")
        except Exception as e:
            raise Exception(f"Database error updating prediction: {str(e)}")
    
    async def get_prediction(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a prediction by job_id."""
        try:
            result = self.supabase.table('predictions').select('*').eq('job_id', job_id).execute()
            
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            raise Exception(f"Database error getting prediction: {str(e)}")
    
    async def get_prediction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get prediction history ordered by upload date."""
        try:
            result = self.supabase.table('predictions').select('*').order('upload_date', desc=True).limit(limit).execute()
            return result.data if result.data else []
        except Exception as e:
            raise Exception(f"Database error getting prediction history: {str(e)}")
    
    async def upload_file_to_storage(self, file_path: str, file_name: str) -> str:
        """Upload file to Supabase storage and return public URL."""
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Upload to storage
            result = self.service_client.storage.from_(settings.SUPABASE_STORAGE_BUCKET).upload(
                file_name, file_data
            )
            
            if result:
                # Get public URL
                public_url = self.service_client.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(file_name)
                return public_url
            else:
                raise Exception("Failed to upload file to storage")
        except Exception as e:
            raise Exception(f"Storage error uploading file: {str(e)}")
    
    async def delete_file_from_storage(self, file_name: str) -> bool:
        """Delete file from Supabase storage."""
        try:
            result = self.service_client.storage.from_(settings.SUPABASE_STORAGE_BUCKET).remove([file_name])
            return bool(result)
        except Exception as e:
            print(f"Storage error deleting file: {str(e)}")
            return False
    
    async def get_file_from_storage(self, file_name: str) -> bytes:
        """Download file from Supabase storage."""
        try:
            result = self.service_client.storage.from_(settings.SUPABASE_STORAGE_BUCKET).download(file_name)
            return result
        except Exception as e:
            raise Exception(f"Storage error downloading file: {str(e)}")

# Global database service instance
database_service = None
try:
    if settings.SUPABASE_URL and settings.SUPABASE_KEY:
        database_service = DatabaseService()
except Exception as e:
    print(f"Warning: Failed to initialize database service: {e}")
    print("Running in fallback mode without database integration")
    database_service = None