@echo off
echo 🚀 Starting Visual Document Analysis RAG System
echo ================================================

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Check if .env file exists for backend
if not exist "backend\.env" (
    echo ⚠️  Backend .env file not found. Creating from template...
    copy backend\env.example backend\.env
    echo 📝 Please edit backend\.env with your configuration
)

REM Check if .env file exists for frontend
if not exist "frontend\.env" (
    echo ⚠️  Frontend .env file not found. Creating from template...
    copy frontend\env.example frontend\.env
    echo 📝 Please edit frontend\.env with your configuration
)

echo 🐳 Starting services with Docker Compose...
docker-compose up -d

echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

echo 🔍 Checking service status...

REM Check MongoDB
docker-compose ps mongodb | findstr "Up" >nul
if errorlevel 1 (
    echo ❌ MongoDB failed to start
) else (
    echo ✅ MongoDB is running
)

REM Check Backend
docker-compose ps backend | findstr "Up" >nul
if errorlevel 1 (
    echo ❌ Backend failed to start
) else (
    echo ✅ Backend is running
)

REM Check Frontend
docker-compose ps frontend | findstr "Up" >nul
if errorlevel 1 (
    echo ❌ Frontend failed to start
) else (
    echo ✅ Frontend is running
)

echo.
echo 🌐 Application URLs:
echo    Frontend: http://localhost:3000
echo    Backend API: http://localhost:8000
echo    API Documentation: http://localhost:8000/docs
echo.
echo 📋 Useful commands:
echo    View logs: docker-compose logs -f
echo    Stop services: docker-compose down
echo    Restart services: docker-compose restart
echo.
echo 🎉 Setup complete! Open http://localhost:3000 in your browser.
pause

