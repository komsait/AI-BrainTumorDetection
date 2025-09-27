"""
Structured JSON logging utilities for the brain tumor detection application.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import traceback

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)

class PredictionLogger:
    """Logger for prediction-related events."""
    
    def __init__(self, logger_name: str = "prediction"):
        self.logger = logging.getLogger(logger_name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the prediction logger."""
        if not self.logger.handlers:
            # Create console handler
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
    
    def log_prediction_start(self, filename: str, file_size: int) -> None:
        """Log the start of a prediction request."""
        extra_fields = {
            "event": "prediction_start",
            "filename": filename,
            "file_size_bytes": file_size,
            "request_id": f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        }
        
        self.logger.info(f"Starting prediction for file: {filename}", 
                        extra={'extra_fields': extra_fields})
    
    def log_classifier_result(self, filename: str, prediction: str, confidence: float, 
                            processing_time: float) -> None:
        """Log classifier prediction result."""
        extra_fields = {
            "event": "classifier_result",
            "filename": filename,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "processing_time_ms": round(processing_time * 1000, 2)
        }
        
        self.logger.info(f"Classifier prediction: {prediction} (confidence: {confidence:.4f})", 
                        extra={'extra_fields': extra_fields})
    
    def log_segmentation_result(self, filename: str, tumor_area_ratio: float, 
                              processing_time: float) -> None:
        """Log segmentation result."""
        extra_fields = {
            "event": "segmentation_result",
            "filename": filename,
            "tumor_area_ratio": round(tumor_area_ratio, 4),
            "processing_time_ms": round(processing_time * 1000, 2)
        }
        
        self.logger.info(f"Segmentation completed - tumor area ratio: {tumor_area_ratio:.4f}", 
                        extra={'extra_fields': extra_fields})
    
    def log_prediction_complete(self, filename: str, final_result: str, 
                              total_time: float, success: bool = True) -> None:
        """Log completion of prediction pipeline."""
        extra_fields = {
            "event": "prediction_complete",
            "filename": filename,
            "final_result": final_result,
            "total_processing_time_ms": round(total_time * 1000, 2),
            "success": success
        }
        
        self.logger.info(f"Prediction completed: {final_result} in {total_time:.2f}s", 
                        extra={'extra_fields': extra_fields})
    
    def log_error(self, filename: str, error_type: str, error_message: str, 
                 processing_stage: str) -> None:
        """Log prediction errors."""
        extra_fields = {
            "event": "prediction_error",
            "filename": filename,
            "error_type": error_type,
            "error_message": error_message,
            "processing_stage": processing_stage
        }
        
        self.logger.error(f"Error in {processing_stage}: {error_message}", 
                         extra={'extra_fields': extra_fields})

def setup_application_logging(log_level: str = "INFO") -> None:
    """Setup application-wide logging configuration."""
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    logging.info("Application logging configured", 
                extra={'extra_fields': {
                    'event': 'logging_setup',
                    'log_level': log_level.upper()
                }})

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with JSON formatting."""
    logger = logging.getLogger(name)
    
    # Ensure JSON formatting if no handlers are present
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    return logger

# Create global prediction logger instance
prediction_logger = PredictionLogger()
