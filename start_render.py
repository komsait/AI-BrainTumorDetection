#!/usr/bin/env python3
"""
Startup script for Render deployment.
Configures environment for CPU-only TensorFlow execution.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_environment():
    """Configure environment variables for CPU-only execution."""
    # Disable CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # TensorFlow CPU configuration
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
    
    # Disable TensorFlow GPU memory growth
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    logger.info("Environment configured for CPU-only execution")

if __name__ == "__main__":
    configure_environment()
    
    # Import and run the FastAPI app
    from app.main import app
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
