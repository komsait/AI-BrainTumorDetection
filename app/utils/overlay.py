"""
Utilities for creating overlay visualizations of segmentation masks.
"""

import numpy as np
import cv2
from PIL import Image
import io
import base64
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def create_overlay_image(original_image: np.ndarray, mask: np.ndarray, 
                        overlay_color: Tuple[int, int, int] = (255, 0, 0), 
                        alpha: float = 0.4) -> np.ndarray:
    """
    Create an overlay of the segmentation mask on the original image.
    
    Args:
        original_image: Original MRI image (RGB format)
        mask: Binary segmentation mask
        overlay_color: RGB color for the overlay (default: red)
        alpha: Transparency level for overlay (0.0 to 1.0)
        
    Returns:
        Image with overlay visualization
    """
    try:
        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        
        # Resize mask to match original image dimensions
        if mask.shape[:2] != original_image.shape[:2]:
            mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        
        # Create colored mask
        colored_mask = np.zeros_like(original_image)
        colored_mask[:, :, 0] = overlay_color[0] * mask_resized  # Red channel
        colored_mask[:, :, 1] = overlay_color[1] * mask_resized  # Green channel
        colored_mask[:, :, 2] = overlay_color[2] * mask_resized  # Blue channel
        
        # Create overlay by blending original image with colored mask
        overlay_image = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)
        
        logger.info(f"Created overlay image with shape: {overlay_image.shape}")
        return overlay_image
        
    except Exception as e:
        logger.error(f"Failed to create overlay image: {str(e)}")
        raise

def create_side_by_side_comparison(original_image: np.ndarray, mask: np.ndarray, 
                                 overlay_image: np.ndarray) -> np.ndarray:
    """
    Create a side-by-side comparison of original image, mask, and overlay.
    
    Args:
        original_image: Original MRI image
        mask: Binary segmentation mask
        overlay_image: Image with overlay visualization
        
    Returns:
        Combined image showing all three views
    """
    try:
        # Resize all images to the same height
        target_height = original_image.shape[0]
        
        # Resize mask to match original image dimensions
        if mask.shape[:2] != original_image.shape[:2]:
            mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        
        # Convert mask to 3-channel for visualization
        mask_3channel = np.stack([mask_resized * 255] * 3, axis=-1)
        
        # Resize overlay to match
        overlay_resized = cv2.resize(overlay_image, (original_image.shape[1], original_image.shape[0]))
        
        # Create side-by-side comparison
        comparison = np.hstack([original_image, mask_3channel, overlay_resized])
        
        logger.info(f"Created side-by-side comparison with shape: {comparison.shape}")
        return comparison
        
    except Exception as e:
        logger.error(f"Failed to create side-by-side comparison: {str(e)}")
        raise

def image_to_base64(image: np.ndarray, format: str = 'PNG') -> str:
    """
    Convert numpy image array to base64 string.
    
    Args:
        image: Image array
        format: Image format ('PNG' or 'JPEG')
        
    Returns:
        Base64 encoded image string
    """
    try:
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        
        # Encode to base64
        image_bytes = buffer.getvalue()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        # Add data URI prefix
        mime_type = 'image/png' if format == 'PNG' else 'image/jpeg'
        data_uri = f"data:{mime_type};base64,{base64_string}"
        
        logger.info(f"Successfully converted image to base64 ({len(base64_string)} characters)")
        return data_uri
        
    except Exception as e:
        logger.error(f"Failed to convert image to base64: {str(e)}")
        raise

def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to numpy image array.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Image as numpy array
    """
    try:
        # Remove data URI prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        logger.info(f"Successfully converted base64 to image with shape: {image_array.shape}")
        return image_array
        
    except Exception as e:
        logger.error(f"Failed to convert base64 to image: {str(e)}")
        raise

def create_contour_visualization(original_image: np.ndarray, mask: np.ndarray, 
                               contour_color: Tuple[int, int, int] = (0, 255, 0), 
                               thickness: int = 2) -> np.ndarray:
    """
    Create visualization with contours drawn on the original image.
    
    Args:
        original_image: Original MRI image
        mask: Binary segmentation mask
        contour_color: BGR color for contours
        thickness: Contour line thickness
        
    Returns:
        Image with contours drawn
    """
    try:
        # Resize mask to match original image dimensions
        if mask.shape[:2] != original_image.shape[:2]:
            mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        
        # Find contours
        contours, _ = cv2.findContours(mask_resized.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy of the original image
        contour_image = original_image.copy()
        
        # Draw contours
        cv2.drawContours(contour_image, contours, -1, contour_color, thickness)
        
        logger.info(f"Drew {len(contours)} contours on image")
        return contour_image
        
    except Exception as e:
        logger.error(f"Failed to create contour visualization: {str(e)}")
        raise
