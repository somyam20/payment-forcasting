#!/usr/bin/env python3
"""
FastAPI application entry point for Meeting Document Generator API
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.backend.routes import document_routes

# Create FastAPI app
app = FastAPI(
    title="MDoc API",
    version="1.0",
    description="Meeting Document Generator API - Transform meeting recordings into professional documents",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(
    document_routes.router,
    prefix="/api/document",
    tags=["Document"]
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MDoc API - Meeting Document Generator",
        "version": "1.0",
        "docs": "/docs",
        "health": "/api/document/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

