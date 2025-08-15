# Visual Document Analysis RAG System

A comprehensive document analysis and question-answering system that combines OCR, vector search, and AI-powered text generation to provide intelligent insights from uploaded documents.

**Created by Farhan Farooq**

## 📋 Project Status

✅ **Fully Functional** - The application is currently running and tested with:
- Document upload and processing
- OCR text extraction from images and PDFs
- Vector-based semantic search
- AI-powered question answering
- Real-time API communication between frontend and backend

## 🚀 Features

- **Multi-format Document Support**: Upload PDFs and images (PNG, JPEG)
- **Advanced OCR Processing**: Extract text from images using EasyOCR
- **Vector-based Search**: Semantic document search using ChromaDB and sentence transformers
- **AI-Powered Q&A**: Generate intelligent answers using Hugging Face models
- **Real-time Processing**: Fast document processing and indexing
- **Modern Web Interface**: Clean, responsive React frontend
- **Document Management**: Upload, view, and delete documents
- **Security Features**: File size validation, type checking, and input sanitization

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance web framework
- **MongoDB**: Document storage with Motor async driver
- **ChromaDB**: Vector database for semantic search
- **Sentence Transformers**: Text embeddings for similarity search
- **EasyOCR**: Optical character recognition
- **PyMuPDF**: PDF processing and text extraction
- **Hugging Face API**: AI text generation

### Frontend
- **React**: Modern UI framework
- **Tailwind CSS**: Utility-first styling
- **Axios**: HTTP client for API communication
- **Radix UI**: Accessible component library (available for future use)

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB (local or cloud)
- Hugging Face API token (optional, for text generation features)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd NerveSparks-main
```

### 2. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your configuration
```

**Environment Variables:**
```bash
# MongoDB Configuration
MONGO_URL=mongodb://localhost:27017
DB_NAME=document_rag_db

# Hugging Face API Token (optional - for text generation features)
HF_TOKEN=your_huggingface_token_here

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

**Start the Backend:**
```bash
# Option 1: Using the startup script
python start.py

# Option 2: Using uvicorn directly
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
yarn install

# Set up environment variables
cp env.example .env
# Edit .env with your backend URL
```

**Environment Variables:**
```bash
# Backend API URL
REACT_APP_BACKEND_URL=http://localhost:8000
```

**Start the Frontend:**
```bash
yarn start
```

### 4. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 🎯 Current Working Setup

The application has been successfully tested and is currently running with:

- **Backend**: FastAPI server on port 8000 with MongoDB Atlas integration
- **Frontend**: React development server on port 3000
- **Database**: MongoDB Atlas cloud database
- **OCR**: EasyOCR models loaded and functional
- **Vector Search**: ChromaDB working with sentence transformers
- **AI Models**: Hugging Face integration (with fallback for missing tokens)

**Tested Features:**
- ✅ Document upload (PDF, PNG, JPEG)
- ✅ OCR text extraction
- ✅ Vector indexing and search
- ✅ AI-powered question answering
- ✅ Real-time API communication
- ✅ Error handling and validation

## 📚 API Endpoints

### Core Endpoints

- `POST /api/upload-document`: Upload and process documents
  - Supports: PDF, PNG, JPEG
  - Max file size: 50MB
  - Returns: Processing results and document ID

- `POST /api/query`: Ask questions about uploaded documents
  - Input: Query text and max results (1-20)
  - Returns: AI-generated answer with source references

- `GET /api/documents`: List all uploaded documents
  - Returns: Document metadata and statistics

- `DELETE /api/documents/{id}`: Delete a document
  - Removes from both database and vector index

- `GET /api/health`: System health check
  - Returns: System status and component health

### Request/Response Examples

**Upload Document:**
```bash
curl -X POST "http://localhost:8000/api/upload-document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Query Documents:**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "max_results": 5}'
```

## 🔧 Configuration

### MongoDB Setup

1. **Local MongoDB:**
   ```bash
   # Install MongoDB
   # Start MongoDB service
   mongod
   ```

2. **MongoDB Atlas (Cloud):**
   - Create a cluster at https://cloud.mongodb.com
   - Get your connection string
   - Update `MONGO_URL` in your `.env` file

### Hugging Face API Setup

1. Create an account at https://huggingface.co
2. Generate an API token
3. Add the token to your `.env` file as `HF_TOKEN`

**Note:** The system works without HF_TOKEN, but text generation features will be limited.

## 🧪 Testing

The system includes comprehensive error handling and validation:

- File type validation
- File size limits (50MB)
- Query length limits (1000 characters)
- Input sanitization
- Graceful error handling

## 🚀 Deployment

### Backend Deployment

1. **Docker (Recommended):**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Cloud Platforms:**
   - **Heroku**: Use the provided `Procfile`
   - **AWS**: Deploy to EC2 or use Lambda
   - **Google Cloud**: Use Cloud Run or App Engine

### Frontend Deployment

1. **Build the application:**
   ```bash
   yarn build
   ```

2. **Deploy to:**
   - **Netlify**: Drag and drop the `build` folder
   - **Vercel**: Connect your repository
   - **AWS S3**: Upload the `build` folder

## 🔒 Security Features

- File type validation
- File size limits
- Input sanitization
- CORS configuration
- Error message sanitization
- Rate limiting (configurable)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

If you encounter any issues:

1. Check the system health endpoint: `GET /api/health`
2. Review the logs for error messages
3. Ensure all environment variables are set correctly
4. Verify MongoDB is running and accessible

## 👨‍💻 About the Creator

**Farhan Farooq** - Full-stack developer and AI enthusiast who created this comprehensive document analysis system. The project demonstrates modern web development practices, AI integration, and robust system architecture.

**Contact:** For questions or collaboration opportunities, please reach out through the project repository.

## 🔄 Changelog

### v1.1.0 (Current)
- ✅ Fixed frontend-backend communication issues
- ✅ Resolved environment variable configuration
- ✅ Added fallback backend URL for reliability
- ✅ Successfully tested document upload and query functionality
- ✅ Verified OCR processing with EasyOCR models
- ✅ Confirmed vector search and AI Q&A working

### v1.0.0
- Initial release
- PDF and image processing
- OCR text extraction
- Vector-based search
- AI-powered Q&A
- Modern web interface
