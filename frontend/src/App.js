import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const API = `${BACKEND_URL}/api`;

const DocumentUpload = ({ onUploadSuccess, isUploading, setIsUploading }) => {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file) => {
    // Validate file type
    const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg'];
    if (!allowedTypes.includes(file.type)) {
      alert('Please upload a PDF or image file (PNG, JPEG)');
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/upload-document`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        onUploadSuccess(response.data);
      }
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Upload failed: ' + (error.response?.data?.detail || 'Unknown error'));
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="upload-section">
      <div
        className={`upload-area ${dragActive ? 'drag-active' : ''} ${isUploading ? 'uploading' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {isUploading ? (
          <div className="upload-progress">
            <div className="spinner"></div>
            <p>Processing document...</p>
          </div>
        ) : (
          <>
            <div className="upload-icon">📄</div>
            <h3>Upload Document</h3>
            <p>Drag and drop your PDF or image file here, or click to browse</p>
            <input
              type="file"
              id="file-input"
              accept=".pdf,image/*"
              onChange={handleFileInput}
              style={{ display: 'none' }}
            />
            <button
              type="button"
              className="browse-button"
              onClick={() => document.getElementById('file-input').click()}
            >
              Browse Files
            </button>
            <div className="supported-formats">
              <small>Supported formats: PDF, PNG, JPEG</small>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const QueryInterface = ({ documents, isQuerying, setIsQuerying }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsQuerying(true);
    try {
      const response = await axios.post(`${API}/query`, {
        query: query.trim(),
        max_results: 5
      });

      setResults(response.data);
    } catch (error) {
      console.error('Query failed:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error';
      alert('Query failed: ' + errorMessage);
      setResults(null); // Clear previous results on error
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="query-section">
      <h2>Ask Questions About Your Documents</h2>
      
      {documents.length === 0 ? (
        <div className="no-documents">
          <p>Upload documents first to start asking questions</p>
        </div>
      ) : (
        <>
          <form onSubmit={handleQuery} className="query-form">
            <div className="query-input-group">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question about your documents..."
                className="query-input"
                disabled={isQuerying}
              />
              <button
                type="submit"
                disabled={isQuerying || !query.trim()}
                className="query-button"
              >
                {isQuerying ? 'Searching...' : 'Ask'}
              </button>
            </div>
          </form>

          {results && (
            <div className="results-section">
              <div className="answer-section">
                <h3>Answer</h3>
                <div className="answer-content">
                  {results.answer}
                </div>
                <div className="answer-meta">
                  <small>
                    Processing time: {results.processing_time_ms}ms | 
                    Sources found: {results.total_sources}
                  </small>
                </div>
              </div>

              {results.source_documents && results.source_documents.length > 0 && (
                <div className="sources-section">
                  <h3>Source Documents</h3>
                  {results.source_documents.map((doc, index) => (
                    <div key={index} className="source-item">
                      <div className="source-header">
                        <span className="source-rank">#{doc.rank}</span>
                        <span className="source-type">{doc.metadata.content_type}</span>
                        <span className="source-page">Page {doc.metadata.page}</span>
                        <span className="source-score">
                          Relevance: {(doc.score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="source-content">
                        {doc.content.length > 300 
                          ? `${doc.content.substring(0, 300)}...`
                          : doc.content
                        }
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};

const DocumentList = ({ documents, onDeleteDocument }) => {
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString() + ' ' + 
           new Date(dateString).toLocaleTimeString();
  };

  return (
    <div className="documents-section">
      <h2>Uploaded Documents ({documents.length})</h2>
      
      {documents.length === 0 ? (
        <div className="no-documents">
          <p>No documents uploaded yet</p>
        </div>
      ) : (
        <div className="documents-list">
          {documents.map((doc) => (
            <div key={doc.id} className="document-item">
              <div className="document-info">
                <div className="document-name">{doc.filename}</div>
                <div className="document-meta">
                  <span>Type: {doc.content_type}</span>
                  <span>Size: {formatFileSize(doc.file_size)}</span>
                  <span>Uploaded: {formatDate(doc.upload_time)}</span>
                </div>
              </div>
              <button
                onClick={() => onDeleteDocument(doc.id)}
                className="delete-button"
                title="Delete document"
              >
                🗑️
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const App = () => {
  const [documents, setDocuments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [systemStatus, setSystemStatus] = useState({ status: 'checking' });

  useEffect(() => {
    checkSystemHealth();
    loadDocuments();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await axios.get(`${API}/health`);
      setSystemStatus(response.data);
    } catch (error) {
      setSystemStatus({ status: 'error', error: 'System unavailable' });
    }
  };

  const loadDocuments = async () => {
    try {
      const response = await axios.get(`${API}/documents`);
      setDocuments(response.data.documents);
    } catch (error) {
      console.error('Failed to load documents:', error);
    }
  };

  const handleUploadSuccess = (uploadResult) => {
    console.log('Upload successful:', uploadResult);
    loadDocuments(); // Refresh document list
    // Show success message
    alert(`Document "${uploadResult.filename}" uploaded successfully!`);
  };

  const handleDeleteDocument = async (documentId) => {
    if (!window.confirm('Are you sure you want to delete this document?')) {
      return;
    }

    try {
      await axios.delete(`${API}/documents/${documentId}`);
      loadDocuments(); // Refresh document list
    } catch (error) {
      console.error('Delete failed:', error);
      alert('Failed to delete document');
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>📄 Visual Document Analysis RAG</h1>
        <p>Upload documents and ask intelligent questions about their content</p>
        <div className={`system-status ${systemStatus.status}`}>
          Status: {systemStatus.status === 'healthy' ? '✅ System Ready' : 
                  systemStatus.status === 'checking' ? '⏳ Checking...' : 
                  '❌ System Error'}
        </div>
      </header>

      <main className="main-content">
        <div className="content-grid">
          <div className="left-panel">
            <DocumentUpload
              onUploadSuccess={handleUploadSuccess}
              isUploading={isUploading}
              setIsUploading={setIsUploading}
            />
            
            <DocumentList
              documents={documents}
              onDeleteDocument={handleDeleteDocument}
            />
          </div>

          <div className="right-panel">
            <QueryInterface
              documents={documents}
              isQuerying={isQuerying}
              setIsQuerying={setIsQuerying}
            />
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Powered by Hugging Face API, EasyOCR, and ChromaDB</p>
      </footer>
    </div>
  );
};

export default App;