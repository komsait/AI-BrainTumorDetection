"""
Image preprocessing utilities for MRI brain tumor detection.
"""

import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def validate_image_file(file_content: bytes, filename: str) -> bool:
    """
    Validate that the uploaded file is a valid image.
    
    Args:
        file_content: Raw file content
        filename: Original filename
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        file_ext = filename.lower().split('.')[-1]
        if f'.{file_ext}' not in valid_extensions:
            logger.warning(f"Invalid file extension: {file_ext}")
            return False
        
        # Try to decode the image
        image = Image.open(io.BytesIO(file_content))
        image.verify()
        
        logger.info(f"Successfully validated image: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Image validation failed for {filename}: {str(e)}")
        return False

def load_image_from_bytes(file_content: bytes) -> np.ndarray:
    """
    Load image from bytes and convert to numpy array.
    
    Args:
        file_content: Raw file content
        
    Returns:
        Image as numpy array in RGB format
    """
    try:
        # Load image using PIL
        image = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        logger.info(f"Successfully loaded image with shape: {image_array.shape}")
        return image_array
        
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {str(e)}")
        raise

def resize_image_for_display(image: np.ndarray, max_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Resize image for display purposes while maintaining aspect ratio.
    
    Args:
        image: Input image array
        max_size: Maximum size for display (width, height)
        
    Returns:
        Resized image array
    """
    try:
        h, w = image.shape[:2]
        max_w, max_h = max_size
        
        # Calculate scaling factor
        scale = min(max_w / w, max_h / h)
        
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize using OpenCV
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {image.shape} to {resized.shape}")
            return resized
        
        return image
        
    except Exception as e:
        logger.error(f"Failed to resize image: {str(e)}")
        raise

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image array
        
    Returns:
        Normalized image array
    """
    try:
        # Ensure image is float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalize to [0, 1]
        normalized = image / 255.0
        
        return normalized
        
    except Exception as e:
        logger.error(f"Failed to normalize image: {str(e)}")
        raise

def enhance_image_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image array
        
    Returns:
        Enhanced image array
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        logger.info("Successfully enhanced image contrast")
        return enhanced
        
    except Exception as e:
        logger.error(f"Failed to enhance image contrast: {str(e)}")
        return image  # Return original if enhancement fails
