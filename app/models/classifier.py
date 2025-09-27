"""
Brain tumor classifier model loader and inference utilities.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Tuple
import os

# Force CPU usage for deployment environments without GPU
try:
    tf.config.set_visible_devices([], 'GPU')
    logger.info("Configured TensorFlow to use CPU only")
except RuntimeError as e:
    logger.warning(f"Could not configure GPU visibility: {e}")
    # This is expected in some deployment environments

logger = logging.getLogger(__name__)

class BrainTumorClassifier:
    """Brain tumor classifier model wrapper."""
    
    def __init__(self, model_path: str = "app/models/brain_tumor_classifier_tuned.keras"):
        """
        Initialize the classifier model.
        
        Args:
            model_path: Path to the trained classifier model
        """
        self.model_path = model_path
        self.model = None
        self.input_shape = (224, 224, 3)
        self.load_model()
    
    def load_model(self) -> None:
        """Load the classifier model from disk."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Successfully loaded classifier model from {self.model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load classifier model: {str(e)}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for classifier inference.
        
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
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict tumor presence in the image.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (prediction_label, confidence_score)
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            confidence = float(prediction[0][0])
            
            # Convert to binary prediction
            prediction_label = "Tumor" if confidence > 0.5 else "No Tumor"
            
            logger.info(f"Classifier prediction: {prediction_label} (confidence: {confidence:.4f})")
            
            return prediction_label, confidence
            
        except Exception as e:
            logger.error(f"Error during classifier prediction: {str(e)}")
            raise
    
    def is_tumor_present(self, image: np.ndarray) -> bool:
        """
        Check if tumor is present in the image.
        
        Args:
            image: Input image array
            
        Returns:
            True if tumor is detected, False otherwise
        """
        prediction, _ = self.predict(image)
        return prediction == "Tumor"
