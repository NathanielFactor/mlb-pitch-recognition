from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, predict
from app.core.config import settings

app = FastAPI(
    title="MLB Pitch Recognition API",
    description="API for classifying MLB pitch types from video clips",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(predict.router, prefix="/api", tags=["predict"])

@app.get("/")
async def root():
    return {"message": "MLB Pitch Recognition API", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "mock_mode": settings.MOCK_MODE}