#!/usr/bin/env python3
"""
Startup script for the Visual Document Analysis RAG System
"""

import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    print("=" * 60)
    print("Visual Document Analysis RAG System")
    print("=" * 60)
    print(f"Starting server on {host}:{port}")
    print("Make sure you have set up your environment variables:")
    print("- MONGO_URL: MongoDB connection string")
    print("- DB_NAME: Database name")
    print("- HF_TOKEN: Hugging Face API token (optional)")
    print("- CORS_ORIGINS: Allowed CORS origins")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
