"""
FastAPI application for brain tumor detection using Keras models.
"""

import time
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io

# Import our custom modules
from app.models.classifier import BrainTumorClassifier
from app.models.segmentation import BrainTumorSegmenter
from app.utils.preprocessing import validate_image_file, load_image_from_bytes
from app.utils.overlay import create_overlay_image, image_to_base64
from app.utils.logger import setup_application_logging, prediction_logger, get_logger

# Setup logging
setup_application_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for detecting and segmenting brain tumors in MRI images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
import os
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static")), name="static")

# Global model instances (loaded at startup)
classifier_model: Optional[BrainTumorClassifier] = None
segmentation_model: Optional[BrainTumorSegmenter] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models at application startup."""
    global classifier_model, segmentation_model
    
    try:
        logger.info("Starting application initialization...")
        
        # Configure TensorFlow for CPU-only execution
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Load classifier model
        logger.info("Loading classifier model...")
        classifier_model = BrainTumorClassifier()
        
        # Load segmentation model
        logger.info("Loading segmentation model...")
        segmentation_model = BrainTumorSegmenter()
        
        logger.info("Application initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "models_loaded": classifier_model is not None and segmentation_model is not None}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main frontend page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/classification", response_class=HTMLResponse)
async def classification_page(request: Request):
    """Serve the classification-only page."""
    return templates.TemplateResponse("classification.html", {"request": request})

@app.get("/segmentation", response_class=HTMLResponse)
async def segmentation_page(request: Request):
    """Serve the segmentation-only page."""
    return templates.TemplateResponse("segmentation.html", {"request": request})


@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    """
    Predict brain tumor presence and segment if detected.
    
    Args:
        file: Uploaded MRI image file
        
    Returns:
        JSON response with prediction results and optional overlay image
    """
    start_time = time.time()
    
    try:
        # Log prediction start
        file_content = await file.read()
        prediction_logger.log_prediction_start(file.filename, len(file_content))
        
        # Validate file
        if not validate_image_file(file_content, file.filename):
            raise HTTPException(
                status_code=400, 
                detail="Invalid image file. Please upload a JPG, JPEG, PNG, or TIF file."
            )
        
        # Load and preprocess image
        image = load_image_from_bytes(file_content)
        logger.info(f"Loaded image with shape: {image.shape}")
        
        # Run classifier
        classifier_start = time.time()
        prediction, confidence = classifier_model.predict(image)
        classifier_time = time.time() - classifier_start
        
        prediction_logger.log_classifier_result(
            file.filename, prediction, confidence, classifier_time
        )
        
        # Prepare response
        response = {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "filename": file.filename
        }
        
        # If tumor detected, run segmentation
        if prediction == "Tumor":
            logger.info("Tumor detected, running segmentation...")
            
            segmentation_start = time.time()
            mask = segmentation_model.predict_mask(image)
            segmentation_time = time.time() - segmentation_start
            
            # Calculate tumor area ratio
            tumor_area_ratio = segmentation_model.get_tumor_area_ratio(mask)
            
            prediction_logger.log_segmentation_result(
                file.filename, tumor_area_ratio, segmentation_time
            )
            
            # Create overlay visualization
            overlay_image = create_overlay_image(image, mask)
            
            # Convert overlay to base64
            overlay_base64 = image_to_base64(overlay_image)
            
            # Add segmentation results to response
            response.update({
                "tumor_area_ratio": round(tumor_area_ratio, 4),
                "overlay_image": overlay_base64,
                "processing_times": {
                    "classifier_ms": round(classifier_time * 1000, 2),
                    "segmentation_ms": round(segmentation_time * 1000, 2)
                }
            })
        else:
            # No tumor detected
            response["processing_times"] = {
                "classifier_ms": round(classifier_time * 1000, 2)
            }
        
        # Log completion
        total_time = time.time() - start_time
        prediction_logger.log_prediction_complete(
            file.filename, prediction, total_time, success=True
        )
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Prediction failed: {str(e)}"
        
        prediction_logger.log_error(
            file.filename if 'file' in locals() else "unknown",
            type(e).__name__,
            str(e),
            "prediction_pipeline"
        )
        
        prediction_logger.log_prediction_complete(
            file.filename if 'file' in locals() else "unknown",
            "error",
            total_time,
            success=False
        )
        
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/classify")
async def classify_only(file: UploadFile = File(...)):
    """
    Classify brain tumor presence only (no segmentation).
    
    Args:
        file: Uploaded MRI image file
        
    Returns:
        JSON response with classification results only
    """
    start_time = time.time()
    
    try:
        # Log prediction start
        file_content = await file.read()
        prediction_logger.log_prediction_start(file.filename, len(file_content))
        
        # Validate file
        if not validate_image_file(file_content, file.filename):
            raise HTTPException(
                status_code=400, 
                detail="Invalid image file. Please upload a JPG, JPEG, PNG, or TIF file."
            )
        
        # Load and preprocess image
        image = load_image_from_bytes(file_content)
        logger.info(f"Loaded image with shape: {image.shape}")
        
        # Run classifier only
        classifier_start = time.time()
        prediction, confidence = classifier_model.predict(image)
        classifier_time = time.time() - classifier_start
        
        prediction_logger.log_classifier_result(
            file.filename, prediction, confidence, classifier_time
        )
        
        # Prepare response
        response = {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "filename": file.filename,
            "processing_times": {
                "classifier_ms": round(classifier_time * 1000, 2)
            }
        }
        
        # Log completion
        total_time = time.time() - start_time
        prediction_logger.log_prediction_complete(
            file.filename, prediction, total_time, success=True
        )
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Classification failed: {str(e)}"
        
        prediction_logger.log_error(
            file.filename if 'file' in locals() else "unknown",
            type(e).__name__,
            str(e),
            "classification_only"
        )
        
        prediction_logger.log_prediction_complete(
            file.filename if 'file' in locals() else "unknown",
            "error",
            total_time,
            success=False
        )
        
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/segment")
async def segment_only(file: UploadFile = File(...)):
    """
    Perform tumor segmentation only (assumes tumor is present).
    
    Args:
        file: Uploaded MRI image file
        
    Returns:
        JSON response with segmentation results
    """
    start_time = time.time()
    
    try:
        # Log prediction start
        file_content = await file.read()
        prediction_logger.log_prediction_start(file.filename, len(file_content))
        
        # Validate file
        if not validate_image_file(file_content, file.filename):
            raise HTTPException(
                status_code=400, 
                detail="Invalid image file. Please upload a JPG, JPEG, PNG, or TIF file."
            )
        
        # Load and preprocess image
        image = load_image_from_bytes(file_content)
        logger.info(f"Loaded image with shape: {image.shape}")
        
        # Run segmentation directly
        segmentation_start = time.time()
        mask = segmentation_model.predict_mask(image)
        segmentation_time = time.time() - segmentation_start
        
        # Calculate tumor area ratio
        tumor_area_ratio = segmentation_model.get_tumor_area_ratio(mask)
        
        prediction_logger.log_segmentation_result(
            file.filename, tumor_area_ratio, segmentation_time
        )
        
        # Create overlay visualization
        overlay_image = create_overlay_image(image, mask)
        
        # Convert overlay to base64
        overlay_base64 = image_to_base64(overlay_image)
        
        # Prepare response
        response = {
            "prediction": "Tumor",  # Assume tumor for segmentation
            "confidence": 1.0,  # Not applicable for segmentation-only
            "filename": file.filename,
            "tumor_area_ratio": round(tumor_area_ratio, 4),
            "overlay_image": overlay_base64,
            "processing_times": {
                "segmentation_ms": round(segmentation_time * 1000, 2)
            }
        }
        
        # Log completion
        total_time = time.time() - start_time
        prediction_logger.log_prediction_complete(
            file.filename, "segmentation", total_time, success=True
        )
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Segmentation failed: {str(e)}"
        
        prediction_logger.log_error(
            file.filename if 'file' in locals() else "unknown",
            type(e).__name__,
            str(e),
            "segmentation_only"
        )
        
        prediction_logger.log_prediction_complete(
            file.filename if 'file' in locals() else "unknown",
            "error",
            total_time,
            success=False
        )
        
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with JSON logging."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}", 
                  extra={'extra_fields': {
                      'event': 'http_exception',
                      'status_code': exc.status_code,
                      'detail': exc.detail,
                      'path': str(request.url)
                  }})
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}", 
                extra={'extra_fields': {
                    'event': 'unexpected_error',
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                    'path': str(request.url)
                }}, exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (for Render deployment) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
