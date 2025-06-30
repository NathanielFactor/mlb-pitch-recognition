import os
from typing import List

class Settings:
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MLB Pitch Recognition API"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ]
    
    # Mock Mode Configuration
    MOCK_MODE: bool = os.getenv("MOCK_MODE", "true").lower() == "true"
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov"]
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/uploads")
    
    # ML Model Configuration (for future use)
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../Internal Infrastructure/src/models/")
    
    def __init__(self):
        # Ensure upload directory exists
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

settings = Settings()