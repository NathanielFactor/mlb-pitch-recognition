"""
Vercel-compatible FastAPI application entry point.
This file serves as the main handler for Vercel serverless functions.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import os
import sys

# Add the api directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.routers import unified_upload, unified_predict
from app.services.unified_ml_service import ml_service
from app.core.config import settings

# Create FastAPI app
app = FastAPI(
    title="MLB Pitch Recognition API",
    description="API for classifying MLB pitch types from video clips",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"  # Allow all origins for now to debug upload issues
    ],
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include unified routers - no prefix needed since vercel.json routes /api/* to this app
app.include_router(unified_upload.router, tags=["upload"])
app.include_router(unified_predict.router, tags=["predict"])

@app.get("/")
async def root():
    return {"message": "MLB Pitch Recognition API", "version": "1.0.0"}

@app.get("/api/test")
async def test_endpoint():
    return {"message": "API is working!", "routes": "upload should work at /api/upload"}

@app.post("/api/upload-direct")
async def upload_direct_test():
    return {"message": "Direct POST route works!", "method": "POST"}

@app.get("/api/health")
async def health_check():
    # Check environment variables
    env_status = {
        "supabase_url": bool(settings.SUPABASE_URL),
        "supabase_key": bool(settings.SUPABASE_KEY),
        "supabase_service_key": bool(settings.SUPABASE_SERVICE_KEY),
        "storage_bucket": bool(settings.SUPABASE_STORAGE_BUCKET)
    }
    
    # Check database connection
    db_status = "disconnected"
    try:
        from app.services.database_service import database_service
        if database_service:
            db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy", 
        "mock_mode": settings.MOCK_MODE,
        "environment": "production",
        "environment_variables": env_status,
        "database_status": db_status
    }

# Create the handler that Vercel will call
handler = Mangum(app, lifespan="off")