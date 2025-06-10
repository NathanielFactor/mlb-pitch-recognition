# Use the official PyTorch image with CUDA support (if needed)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory in container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port (optional if using an API like FastAPI or Streamlit later)
EXPOSE 8000

# Set default command (can be changed later to run train.py or inference)
CMD ["python", "inference.py", "--video_path", "videos/sample.mp4"]
