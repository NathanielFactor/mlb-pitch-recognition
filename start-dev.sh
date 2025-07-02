#!/bin/bash

# MLB Pitch Recognition - Development Startup Script

echo "🏟️  Starting MLB Pitch Recognition Development Environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Environment variables are configured in docker-compose.dev.yml

# Start development services
echo "🚀 Starting services with Docker Compose..."
docker-compose -f docker/docker-compose.dev.yml up --build

echo "✅ Development environment started!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"