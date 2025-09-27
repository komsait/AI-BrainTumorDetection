"""
Unit test script for the Brain Tumor Detection API.
This script tests the model loading and inference pipeline.
"""

import os
import sys
import numpy as np
import logging
from PIL import Image, ImageDraw
import requests
import json
import time

# Add app directory to path
sys.path.append('app')

from app.models.classifier import BrainTumorClassifier
from app.models.segmentation import BrainTumorSegmenter
from app.utils.preprocessing import validate_image_file, load_image_from_bytes
from app.utils.overlay import create_overlay_image, image_to_base64

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_mri_image(size=(224, 224), has_tumor=False):
    """
    Create a synthetic MRI-like test image.
    
    Args:
        size: Image dimensions (width, height)
        has_tumor: Whether to include a simulated tumor region
        
    Returns:
        PIL Image object
    """
    # Create a grayscale base image with some noise
    image_array = np.random.normal(128, 30, size).astype(np.uint8)
    
    # Add some anatomical-like structures
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Add brain-like circular region
    mask = np.zeros(size, dtype=np.uint8)
    draw = ImageDraw.Draw(Image.fromarray(mask))
    draw.ellipse([center_x-80, center_y-80, center_x+80, center_y+80], fill=255)
    
    # Apply mask
    image_array = np.where(mask > 0, image_array, 50)
    
    # Add tumor if requested
    if has_tumor:
        tumor_x = center_x + 20
        tumor_y = center_y - 10
        draw.ellipse([tumor_x-15, tumor_y-15, tumor_x+15, tumor_y+15], fill=255)
        tumor_mask = np.array(draw._image)
        image_array = np.where(tumor_mask > 0, 200, image_array)
    
    # Convert to RGB
    rgb_image = np.stack([image_array] * 3, axis=-1)
    
    return Image.fromarray(rgb_image)

def test_model_loading():
    """Test that models can be loaded successfully."""
    logger.info("Testing model loading...")
    
    try:
        # Test classifier loading
        classifier = BrainTumorClassifier()
        logger.info("✓ Classifier model loaded successfully")
        
        # Test segmentation loading
        segmenter = BrainTumorSegmenter()
        logger.info("✓ Segmentation model loaded successfully")
        
        return classifier, segmenter
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {str(e)}")
        raise

def test_preprocessing():
    """Test image preprocessing functions."""
    logger.info("Testing preprocessing functions...")
    
    try:
        # Create test image
        test_image = create_test_mri_image()
        
        # Convert to bytes
        import io
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Test validation
        is_valid = validate_image_file(img_bytes.getvalue(), 'test.png')
        assert is_valid, "Image validation should pass for valid PNG"
        logger.info("✓ Image validation works correctly")
        
        # Test loading
        image_array = load_image_from_bytes(img_bytes.getvalue())
        assert image_array.shape[-1] == 3, "Loaded image should be RGB"
        logger.info("✓ Image loading works correctly")
        
    except Exception as e:
        logger.error(f"✗ Preprocessing test failed: {str(e)}")
        raise

def test_classifier_inference(classifier):
    """Test classifier inference."""
    logger.info("Testing classifier inference...")
    
    try:
        # Create test images
        normal_image = create_test_mri_image(has_tumor=False)
        tumor_image = create_test_mri_image(has_tumor=True)
        
        # Convert to numpy arrays
        normal_array = np.array(normal_image)
        tumor_array = np.array(tumor_image)
        
        # Test predictions
        normal_pred, normal_conf = classifier.predict(normal_array)
        tumor_pred, tumor_conf = classifier.predict(tumor_array)
        
        logger.info(f"Normal image prediction: {normal_pred} (confidence: {normal_conf:.4f})")
        logger.info(f"Tumor image prediction: {tumor_pred} (confidence: {tumor_conf:.4f})")
        
        logger.info("✓ Classifier inference completed successfully")
        
        return normal_pred, tumor_pred
        
    except Exception as e:
        logger.error(f"✗ Classifier inference failed: {str(e)}")
        raise

def test_segmentation_inference(segmenter):
    """Test segmentation inference."""
    logger.info("Testing segmentation inference...")
    
    try:
        # Create test image with tumor
        test_image = create_test_mri_image(has_tumor=True)
        test_array = np.array(test_image)
        
        # Test segmentation
        mask = segmenter.predict_mask(test_array)
        
        assert mask.shape == (128, 128, 1), f"Expected mask shape (128, 128, 1), got {mask.shape}"
        logger.info(f"✓ Generated mask with shape: {mask.shape}")
        
        # Test tumor area calculation
        tumor_ratio = segmenter.get_tumor_area_ratio(mask)
        logger.info(f"✓ Tumor area ratio: {tumor_ratio:.4f}")
        
        return mask
        
    except Exception as e:
        logger.error(f"✗ Segmentation inference failed: {str(e)}")
        raise

def test_overlay_creation(image_array, mask):
    """Test overlay creation."""
    logger.info("Testing overlay creation...")
    
    try:
        # Create overlay
        overlay = create_overlay_image(image_array, mask)
        
        assert overlay.shape == image_array.shape, "Overlay should have same shape as input"
        logger.info(f"✓ Created overlay with shape: {overlay.shape}")
        
        # Test base64 conversion
        base64_str = image_to_base64(overlay)
        assert base64_str.startswith('data:image/png;base64,'), "Base64 should start with data URI prefix"
        logger.info(f"✓ Converted overlay to base64 ({len(base64_str)} characters)")
        
    except Exception as e:
        logger.error(f"✗ Overlay creation failed: {str(e)}")
        raise

def test_api_endpoints():
    """Test API endpoints if server is running."""
    logger.info("Testing API endpoints...")
    
    try:
        # Test health endpoint
        response = requests.get('http://localhost:8000/health', timeout=10)
        if response.status_code == 200:
            logger.info("✓ Health endpoint is accessible")
        else:
            logger.warning(f"Health endpoint returned status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        logger.warning("✗ API server not running - skipping endpoint tests")
    except Exception as e:
        logger.error(f"✗ API endpoint test failed: {str(e)}")

def run_full_pipeline_test():
    """Run a complete pipeline test."""
    logger.info("Running full pipeline test...")
    
    try:
        # Create test image with tumor
        test_image = create_test_mri_image(has_tumor=True)
        test_array = np.array(test_image)
        
        # Load models
        classifier = BrainTumorClassifier()
        segmenter = BrainTumorSegmenter()
        
        # Run classifier
        start_time = time.time()
        prediction, confidence = classifier.predict(test_array)
        classifier_time = time.time() - start_time
        
        logger.info(f"Classifier result: {prediction} (confidence: {confidence:.4f}) in {classifier_time:.2f}s")
        
        # Run segmentation if tumor detected
        if prediction == "Tumor":
            start_time = time.time()
            mask = segmenter.predict_mask(test_array)
            segmentation_time = time.time() - start_time
            
            tumor_ratio = segmenter.get_tumor_area_ratio(mask)
            logger.info(f"Segmentation completed in {segmentation_time:.2f}s, tumor ratio: {tumor_ratio:.4f}")
            
            # Create overlay
            overlay = create_overlay_image(test_array, mask)
            base64_overlay = image_to_base64(overlay)
            logger.info(f"Created overlay and converted to base64")
        
        logger.info("✓ Full pipeline test completed successfully")
        
    except Exception as e:
        logger.error(f"✗ Full pipeline test failed: {str(e)}")
        raise

def main():
    """Run all tests."""
    logger.info("Starting Brain Tumor Detection API tests...")
    
    try:
        # Test model loading
        classifier, segmenter = test_model_loading()
        
        # Test preprocessing
        test_preprocessing()
        
        # Test classifier
        normal_pred, tumor_pred = test_classifier_inference(classifier)
        
        # Test segmentation
        test_array = np.array(create_test_mri_image(has_tumor=True))
        mask = test_segmentation_inference(segmenter)
        
        # Test overlay
        test_overlay_creation(test_array, mask)
        
        # Test API endpoints
        test_api_endpoints()
        
        # Run full pipeline test
        run_full_pipeline_test()
        
        logger.info("🎉 All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Tests failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
