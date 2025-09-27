#!/usr/bin/env python3
"""
Development startup script for the Brain Tumor Detection API.
"""

import uvicorn
import sys
import os

# Add app directory to path
sys.path.append('app')

if __name__ == "__main__":
    # Check if models exist
    classifier_path = "app/models/brain_tumor_classifier_tuned.keras"
    segmentation_path = "app/models/final_unet_model.keras"
    
    if not os.path.exists(classifier_path):
        print(f"❌ Classifier model not found: {classifier_path}")
        sys.exit(1)
    
    if not os.path.exists(segmentation_path):
        print(f"❌ Segmentation model not found: {segmentation_path}")
        sys.exit(1)
    
    print("🚀 Starting Brain Tumor Detection API in development mode...")
    print("📱 Web Interface: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("🛑 Press Ctrl+C to stop")
    
    # Run the development server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
        log_level="info"
    )
