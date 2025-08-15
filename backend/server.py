from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import asyncio
import io
import base64
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import easyocr
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
import tempfile
import shutil

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
try:
    mongo_url = os.environ['MONGO_URL']
    db_name = os.environ['DB_NAME']
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
except KeyError as e:
    logging.error(f"Missing required environment variable: {e}")
    raise RuntimeError(f"Missing required environment variable: {e}")

# Initialize Hugging Face API
HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    logging.warning("HF_TOKEN not found in environment variables. Text generation features will be limited.")

# Initialize models and services
embedding_model = None
ocr_reader = None
chroma_client = None
collection = None

class DocumentProcessor:
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
    async def process_pdf(self, file_content: bytes) -> Dict[str, Any]:
        """Extract text and images from PDF"""
        try:
            # Create temporary file for PDF processing
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                
                # Open PDF with PyMuPDF
                doc = fitz.open(temp_file.name)
                extracted_content = {
                    "text_content": [],
                    "images": [],
                    "tables": [],
                    "metadata": {"pages": doc.page_count, "type": "pdf"}
                }
                
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    
                    # Extract text
                    text = page.get_text()
                    if text.strip():
                        extracted_content["text_content"].append({
                            "page": page_num + 1,
                            "content": text,
                            "type": "text"
                        })
                    
                    # Extract images
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            if pix.n < 5:  # GRAY or RGB
                                img_data = pix.tobytes("png")
                                img_base64 = base64.b64encode(img_data).decode()
                                
                                # OCR on image
                                ocr_text = await self.process_image_ocr(img_data)
                                
                                extracted_content["images"].append({
                                    "page": page_num + 1,
                                    "image_data": img_base64,
                                    "ocr_text": ocr_text,
                                    "type": "image"
                                })
                            pix = None
                        except Exception as e:
                            logging.warning(f"Failed to extract image from page {page_num}: {e}")
                
                doc.close()
                
                # Clean up temporary file
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                
                return extracted_content
            
        except Exception as e:
            logging.error(f"PDF processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
    
    async def process_image_ocr(self, image_data: bytes) -> str:
        """Extract text from image using OCR with optimization for large images"""
        try:
            global ocr_reader
            if ocr_reader is None:
                ocr_reader = easyocr.Reader(['en'])
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Optimize image size for OCR if it's too large
            max_size = (1600, 1600)  # Maximum dimensions for OCR
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = np.array(image)
            
            # Perform OCR with timeout handling
            try:
                results = ocr_reader.readtext(image_array)
            except Exception as ocr_error:
                logging.warning(f"OCR processing failed: {ocr_error}")
                return ""
            
            # Extract text
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Lower threshold for better text extraction
                    extracted_text.append(text.strip())
            
            # Clean and join text
            final_text = " ".join(extracted_text)
            return final_text[:2000]  # Limit text length to avoid memory issues
            
        except Exception as e:
            logging.warning(f"OCR failed: {e}")
            return ""
    
    async def process_image_file(self, file_content: bytes) -> Dict[str, Any]:
        """Process standalone image file"""
        try:
            # OCR processing
            ocr_text = await self.process_image_ocr(file_content)
            
            # Convert to base64 for storage
            img_base64 = base64.b64encode(file_content).decode()
            
            return {
                "text_content": [{"page": 1, "content": ocr_text, "type": "ocr"}] if ocr_text else [],
                "images": [{"page": 1, "image_data": img_base64, "ocr_text": ocr_text, "type": "image"}],
                "tables": [],
                "metadata": {"pages": 1, "type": "image"}
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

class RAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.collection = None
        
    async def initialize(self):
        """Initialize embedding model and vector database"""
        global embedding_model, chroma_client, collection
        
        if embedding_model is None:
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        if chroma_client is None:
            chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
            
        if collection is None:
            try:
                collection = chroma_client.get_collection("documents")
            except:
                collection = chroma_client.create_collection("documents")
    
    async def add_document_to_index(self, document_id: str, content: Dict[str, Any]):
        """Add processed document content to vector index"""
        await self.initialize()
        
        chunks = []
        metadatas = []
        ids = []
        
        # Process text content
        for i, text_item in enumerate(content.get("text_content", [])):
            text = text_item["content"]
            if len(text.strip()) > 20:  # Only index substantial text
                chunk_id = f"{document_id}_text_{i}"
                chunks.append(text)
                metadatas.append({
                    "document_id": document_id,
                    "type": "text",
                    "page": text_item.get("page", 1),
                    "content_type": text_item.get("type", "text")
                })
                ids.append(chunk_id)
        
        # Process OCR text from images
        for i, image_item in enumerate(content.get("images", [])):
            ocr_text = image_item.get("ocr_text", "")
            if len(ocr_text.strip()) > 10:  # Only index substantial OCR text
                chunk_id = f"{document_id}_ocr_{i}"
                chunks.append(ocr_text)
                metadatas.append({
                    "document_id": document_id,
                    "type": "ocr",
                    "page": image_item.get("page", 1),
                    "content_type": "image_text"
                })
                ids.append(chunk_id)
        
        if chunks:
            # Generate embeddings
            embeddings = embedding_model.encode(chunks).tolist()
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
    
    async def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search documents using semantic similarity"""
        await self.initialize()
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query]).tolist()
        
        # Search in collection
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "score": 1 - results['distances'][0][i],  # Convert distance to similarity
                "rank": i + 1
            })
        
        return formatted_results
    
    async def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer using Hugging Face API with fallback strategies"""
        try:
            # Prepare context
            context = "\n\n".join([
                f"Document {i+1} (Page {doc['metadata'].get('page', 'Unknown')}): {doc['content'][:400]}..."
                for i, doc in enumerate(context_docs[:3])
            ])
            
            # Try multiple models in order of preference
            models_to_try = [
                "google/flan-t5-base",
                "google/flan-t5-small", 
                "microsoft/DialoGPT-medium",
                "gpt2"
            ]
            
            for model in models_to_try:
                try:
                    # Adjust prompt format based on model type
                    if "flan-t5" in model:
                        prompt = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
                    elif "gpt2" in model or "DialoGPT" in model:
                        prompt = f"Based on the document content, answer this question: {query}\n\nContext:\n{context}\n\nAnswer:"
                    else:
                        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
                    
                    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                    payload = {
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 200,
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "do_sample": True,
                            "return_full_text": False
                        }
                    }
                    
                    response = requests.post(
                        f"https://api-inference.huggingface.co/models/{model}",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get("generated_text", "")
                            if generated_text and len(generated_text.strip()) > 10:
                                # Clean up the response
                                answer = generated_text.strip()
                                # Remove any repeated context
                                if "Context:" in answer:
                                    answer = answer.split("Answer:")[-1].strip()
                                return answer[:500]  # Limit response length
                        
                    elif response.status_code == 503:
                        logging.info(f"Model {model} is loading, trying next model")
                        continue
                    else:
                        logging.warning(f"Model {model} returned {response.status_code}: {response.text}")
                        continue
                        
                except Exception as model_error:
                    logging.warning(f"Error with model {model}: {model_error}")
                    continue
            
            # If all models fail, provide a basic extractive answer
            return self._generate_extractive_answer(query, context_docs)
                
        except Exception as e:
            logging.error(f"Answer generation failed: {e}")
            return self._generate_extractive_answer(query, context_docs)
    
    def _generate_extractive_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate a basic extractive answer when AI generation fails"""
        try:
            query_lower = query.lower()
            
            # Find the most relevant document
            best_doc = None
            best_score = 0
            
            for doc in context_docs[:3]:
                content_lower = doc['content'].lower()
                # Simple keyword matching
                matches = sum(1 for word in query_lower.split() if word in content_lower)
                if matches > best_score:
                    best_score = matches
                    best_doc = doc
            
            if best_doc:
                # Extract relevant sentences
                sentences = best_doc['content'].split('.')
                relevant_sentences = []
                
                for sentence in sentences[:5]:  # Check first 5 sentences
                    sentence_lower = sentence.lower()
                    if any(word in sentence_lower for word in query_lower.split()):
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    answer = '. '.join(relevant_sentences[:2])  # Use top 2 relevant sentences
                    return f"Based on the documents: {answer}."
            
            # Fallback response
            return f"I found information related to your query in the documents. The most relevant content discusses: {context_docs[0]['content'][:200]}..."
        
        except Exception:
            return "I found relevant information in your documents, but I'm having trouble generating a specific answer right now. Please try rephrasing your question."

# Initialize global instances
document_processor = DocumentProcessor()
rag_system = RAGSystem()

# Create the main app
app = FastAPI(title="Visual Document Analysis RAG", version="1.0.0")
api_router = APIRouter(prefix="/api")

# Models
class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    pages_processed: int
    content_extracted: Dict[str, int]
    processing_time_ms: int
    success: bool
    message: str

class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    source_documents: List[Dict[str, Any]]
    processing_time_ms: int
    total_sources: int

class DocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total_count: int

# Routes
@api_router.get("/")
async def root():
    return {"message": "Visual Document Analysis RAG System", "status": "running"}

@api_router.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (PDF, image, etc.)"""
    start_time = datetime.now()
    
    try:
        # Validate file type
        allowed_types = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Supported types: PDF, PNG, JPEG"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 50MB."
            )
        
        # Validate filename
        if not file.filename or len(file.filename) > 255:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename. Filename must be provided and less than 255 characters."
            )
        document_id = str(uuid.uuid4())
        
        # Process based on file type
        if file.content_type == 'application/pdf':
            processed_content = await document_processor.process_pdf(file_content)
        else:  # Image files
            processed_content = await document_processor.process_image_file(file_content)
        
        # Store in database
        document_record = {
            "id": document_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "processed_content": processed_content,
            "upload_time": datetime.utcnow(),
            "file_size": len(file_content)
        }
        
        await db.documents.insert_one(document_record)
        
        # Add to vector index
        await rag_system.add_document_to_index(document_id, processed_content)
        
        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Prepare response
        content_stats = {
            "text_chunks": len(processed_content.get("text_content", [])),
            "images": len(processed_content.get("images", [])),
            "tables": len(processed_content.get("tables", []))
        }
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            pages_processed=processed_content["metadata"]["pages"],
            content_extracted=content_stats,
            processing_time_ms=processing_time,
            success=True,
            message="Document processed and indexed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@api_router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document collection using RAG"""
    start_time = datetime.now()
    
    try:
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty."
            )
        
        if len(request.query) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Query too long. Maximum length is 1000 characters."
            )
        
        # Validate max_results
        if request.max_results and (request.max_results < 1 or request.max_results > 20):
            raise HTTPException(
                status_code=400,
                detail="max_results must be between 1 and 20."
            )
        
        # Search for relevant documents
        search_results = await rag_system.search_documents(
            request.query, 
            n_results=request.max_results
        )
        
        if not search_results:
            return QueryResponse(
                query=request.query,
                answer="I couldn't find any relevant information in the uploaded documents to answer your question.",
                source_documents=[],
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                total_sources=0
            )
        
        # Generate answer using RAG
        answer = await rag_system.generate_answer(request.query, search_results)
        
        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            source_documents=search_results,
            processing_time_ms=processing_time,
            total_sources=len(search_results)
        )
        
    except Exception as e:
        logging.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@api_router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = await db.documents.find(
            {},
            {"id": 1, "filename": 1, "content_type": 1, "upload_time": 1, "file_size": 1, "_id": 0}
        ).to_list(100)
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
        
    except Exception as e:
        logging.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@api_router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its embeddings"""
    try:
        # Delete from database
        result = await db.documents.delete_one({"id": document_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove from vector index (delete all chunks for this document)
        try:
            await rag_system.initialize()
            # Get all IDs for this document
            all_results = collection.get(include=["metadatas"])
            ids_to_delete = [
                id for id, metadata in zip(all_results['ids'], all_results['metadatas'])
                if metadata.get('document_id') == document_id
            ]
            
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
        except Exception as e:
            logging.warning(f"Failed to remove document from vector index: {e}")
        
        return {"message": "Document deleted successfully", "document_id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        await db.documents.count_documents({})
        
        # Check if models are loaded
        model_status = "loaded" if embedding_model else "not loaded"
        
        return {
            "status": "healthy",
            "database": "connected",
            "embedding_model": model_status,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }

# Include router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Initializing RAG system...")
    await rag_system.initialize()
    logger.info("RAG system initialized successfully")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()