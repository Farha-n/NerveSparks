#!/bin/bash

echo "🚀 Starting Visual Document Analysis RAG System"
echo "================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists for backend
if [ ! -f "backend/.env" ]; then
    echo "⚠️  Backend .env file not found. Creating from template..."
    cp backend/env.example backend/.env
    echo "📝 Please edit backend/.env with your configuration"
fi

# Check if .env file exists for frontend
if [ ! -f "frontend/.env" ]; then
    echo "⚠️  Frontend .env file not found. Creating from template..."
    cp frontend/env.example frontend/.env
    echo "📝 Please edit frontend/.env with your configuration"
fi

echo "🐳 Starting services with Docker Compose..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service status..."

# Check MongoDB
if docker-compose ps mongodb | grep -q "Up"; then
    echo "✅ MongoDB is running"
else
    echo "❌ MongoDB failed to start"
fi

# Check Backend
if docker-compose ps backend | grep -q "Up"; then
    echo "✅ Backend is running"
else
    echo "❌ Backend failed to start"
fi

# Check Frontend
if docker-compose ps frontend | grep -q "Up"; then
    echo "✅ Frontend is running"
else
    echo "❌ Frontend failed to start"
fi

echo ""
echo "🌐 Application URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart services: docker-compose restart"
echo ""
echo "🎉 Setup complete! Open http://localhost:3000 in your browser."

