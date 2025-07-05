from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Simple test API working"}

@app.get("/test")
def test():
    return {"status": "success", "message": "GET endpoint working"}

@app.post("/upload")
def upload():
    return {"status": "success", "message": "POST endpoint working", "method": "POST"}

# Handler for Vercel
handler = Mangum(app)