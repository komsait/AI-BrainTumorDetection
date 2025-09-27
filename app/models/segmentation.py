"""
Brain tumor segmentation model loader and inference utilities.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Tuple
import os

logger = logging.getLogger(__name__)

# Force CPU usage for deployment environments without GPU
try:
    tf.config.set_visible_devices([], 'GPU')
    logger.info("Configured TensorFlow to use CPU only")
except RuntimeError as e:
    logger.warning(f"Could not configure GPU visibility: {e}")
    # This is expected in some deployment environments

class BrainTumorSegmenter:
    """Brain tumor segmentation model wrapper."""
    
    def __init__(self, model_path: str = "app/models/final_unet_model.keras"):
        """
        Initialize the segmentation model.
        
        Args:
            model_path: Path to the trained segmentation model
        """
        self.model_path = model_path
        self.model = None
        self.input_shape = (128, 128, 3)
        self.output_shape = (128, 128, 1)
        self.load_model()
    
    def load_model(self) -> None:
        """Load the segmentation model from disk."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Define custom objects that might be used in the model
            def dice_coefficient(y_true, y_pred):
                """Dice coefficient metric for segmentation."""
                smooth = 1.0
                y_true_f = tf.keras.backend.flatten(y_true)
                y_pred_f = tf.keras.backend.flatten(y_pred)
                intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
                return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
            
            def dice_loss(y_true, y_pred):
                """Dice loss function."""
                return 1 - dice_coefficient(y_true, y_pred)
            
            # Custom objects dictionary
            custom_objects = {
                'dice_coefficient': dice_coefficient,
                'dice_loss': dice_loss,
            }
            
            # Try to load with custom objects
            try:
                self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
                logger.info(f"Successfully loaded segmentation model from {self.model_path}")
            except Exception as custom_error:
                logger.warning(f"Failed to load with custom objects: {str(custom_error)}")
                logger.info("Attempting to load model without custom objects...")
                
                # Try loading without custom objects (compile=False)
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                logger.info(f"Successfully loaded segmentation model (without compilation) from {self.model_path}")
            
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {str(e)}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for segmentation inference.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        # Resize to model input shape
        image_resized = tf.image.resize(image, self.input_shape[:2])
        
        # Normalize to [0, 1] range
        image_normalized = image_resized / 255.0
        
        # Add batch dimension
        image_batch = tf.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate segmentation mask for the input image.
        
        Args:
            image: Input image array
            
        Returns:
            Binary segmentation mask
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            mask_prediction = self.model.predict(processed_image, verbose=0)
            
            # Remove batch dimension
            mask = mask_prediction[0]
            
            # Apply threshold to create binary mask
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            logger.info(f"Generated segmentation mask with shape: {binary_mask.shape}")
            logger.info(f"Mask statistics - Min: {binary_mask.min()}, Max: {binary_mask.max()}, Mean: {binary_mask.mean():.4f}")
            
            return binary_mask
            
        except Exception as e:
            logger.error(f"Error during segmentation prediction: {str(e)}")
            raise
    
    def get_tumor_area_ratio(self, mask: np.ndarray) -> float:
        """
        Calculate the ratio of tumor area to total image area.
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Ratio of tumor pixels to total pixels
        """
        total_pixels = mask.size
        tumor_pixels = np.sum(mask)
        return float(tumor_pixels / total_pixels)
