# Brain Tumor Detection API

- A production-ready FastAPI web application for detecting and segmenting brain tumors in MRI images using TensorFlow/Keras models.
- can be viewed at: :https://aitumor.tahoonkhaled.com

## Features

- **Brain Tumor Classification**: Detects presence of tumors in MRI images
- **Tumor Segmentation**: Creates precise masks of tumor regions when detected
- **Web Interface**: Beautiful, responsive frontend for easy image upload
- **REST API**: Complete API for integration with other applications
- **Real-time Visualization**: Overlay tumor masks on original images
- **Production Ready**: Dockerized with structured logging and error handling


## Resources
- Download original classification dataset from Kaggle:https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
- Download original segmentation dataset from Kaggle:https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- Download Notebooks:https://drive.google.com/drive/folders/1G8kZCRk5CwmNAjCcPdneYt48Qc24UUu4?usp=sharing


## Architecture

The application uses a two-stage pipeline:

1. **Classifier Model** (`brain_tumor_classifier_tuned.keras`): Predicts if a tumor is present
2. **Segmentation Model** (`final_unet_model.keras`): Creates detailed tumor masks

### Workflow

1. User uploads an MRI image (JPG, PNG, JPEG)
2. Classifier determines tumor presence
3. If **no tumor**: Returns `{"prediction": "No Tumor"}`
4. If **tumor detected**: 
   - Runs segmentation to create mask
   - Overlays mask on original image
   - Returns JSON with prediction and base64-encoded overlay image

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/              # Model loading utilities
│   │   ├── classifier.py    # Brain tumor classifier
│   │   ├── segmentation.py  # Tumor segmentation model
│   │   ├── brain_tumor_classifier_tuned.keras
│   │   └── final_unet_model.keras
│   └── utils/               # Utility functions
│       ├── preprocessing.py # Image preprocessing
│       ├── overlay.py       # Mask visualization
│       └── logger.py        # Structured logging
├── templates/
│   └── index.html          # Web frontend
├── static/                 # Static assets
├── Dockerfile              # Production container
├── requirements.txt        # Python dependencies
├── test_app.py            # Unit tests
└── README.md              # This file
```

## Quick Start

### Using Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t brain-tumor-detection .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 brain-tumor-detection
   ```

3. **Access the application:**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Run tests:**
   ```bash
   python test_app.py
   ```

## 📡 API Endpoints

### `GET /`
Serves the web interface for image upload.

### `POST /predict`
Upload an MRI image for tumor detection and segmentation.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response (No Tumor):**
```json
{
  "prediction": "No Tumor",
  "confidence": 0.8765,
  "filename": "brain_scan.jpg",
  "processing_times": {
    "classifier_ms": 245.67
  }
}
```

**Response (Tumor Detected):**
```json
{
  "prediction": "Tumor",
  "confidence": 0.9234,
  "filename": "brain_scan.jpg",
  "tumor_area_ratio": 0.1234,
  "overlay_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "processing_times": {
    "classifier_ms": 245.67,
    "segmentation_ms": 567.89
  }
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true
}
```

## Testing

The application includes comprehensive tests:

```bash
# Run all tests
python test_app.py

# Test specific components
python -c "from app.models.classifier import BrainTumorClassifier; BrainTumorClassifier()"
```

Tests cover:
- Model loading and initialization
- Image preprocessing
- Classification inference
- Segmentation inference
- Overlay creation
- API endpoints (if server is running)

## Model Specifications

### Classifier Model
- **Input Shape**: (224, 224, 3)
- **Preprocessing**: Resize to 224x224, normalize to [0, 1]
- **Output**: Binary prediction (Tumor/No Tumor) with confidence score

### Segmentation Model
- **Input Shape**: (128, 128, 3)
- **Preprocessing**: Resize to 128x128, normalize to [0, 1]
- **Output Shape**: (128, 128, 1)
- **Postprocessing**: Binary threshold at 0.5

## Configuration

### Environment Variables
- `PYTHONPATH`: Set to `/app` in container
- `PYTHONUNBUFFERED`: Set to `1` for immediate logging
- `TF_CPP_MIN_LOG_LEVEL`: Set to `2` to reduce TensorFlow logs

### Logging
The application uses structured JSON logging:
- All logs are written to stdout in JSON format
- Includes prediction results, processing times, and errors
- Compatible with log aggregation systems (ELK, Fluentd, etc.)

## Docker Details

### Multi-stage Build
- **Builder stage**: Installs dependencies and builds packages
- **Production stage**: Minimal runtime with only necessary files
- **Security**: Runs as non-root user
- **Health checks**: Built-in health monitoring

### Production Settings
- **Server**: Gunicorn with Uvicorn workers
- **Workers**: 2 worker processes
- **Timeout**: 120 seconds per request
- **Port**: 8000

## Security Features

- Non-root user execution in container
- Input validation for uploaded files
- File type restrictions (JPG, PNG, JPEG only)
- Structured error handling
- No temporary file storage

## Performance

- Models loaded once at startup (not per request)
- Efficient image preprocessing
- Base64 encoding for inline image delivery
- Optimized Docker image size
- Connection pooling and keep-alive
