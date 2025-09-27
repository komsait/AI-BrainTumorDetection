# Render Deployment Guide for AI Brain Tumor Detection

This guide explains how to deploy your AI Brain Tumor Detection application to Render and resolve the common deployment issues.

## Issues Fixed

### 1. CUDA Errors
**Problem**: `CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)`

**Solution**: 
- Added CPU-only configuration for TensorFlow
- Set environment variables to disable CUDA
- Added graceful error handling for GPU initialization

### 2. Port Binding Issues
**Problem**: `No open HTTP ports detected on 0.0.0.0`

**Solution**:
- Modified application to use `PORT` environment variable from Render
- Updated Dockerfile CMD to use dynamic port binding
- Fixed health check to use correct port

## Deployment Steps

### 1. Render Service Configuration

When creating your Render service:

1. **Service Type**: Choose "Web Service"
2. **Environment**: Docker
3. **Dockerfile**: Use the provided Dockerfile
4. **Port**: Leave empty (Render will set PORT environment variable)

### 2. Environment Variables

Set these environment variables in your Render dashboard:

```
PORT=8000
PYTHONPATH=/app
TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=
TF_FORCE_GPU_ALLOW_GROWTH=false
```

### 3. Build Configuration

- **Build Command**: `docker build -t brain-tumor-detection .`
- **Start Command**: `python start_render.py`

### 4. Resource Requirements

Recommended Render plan settings:
- **Instance Type**: Starter (1 GB RAM) or Standard (2 GB RAM)
- **CPU**: 1-2 vCPUs
- **Timeout**: 120 seconds

## Files Modified

### `app/main.py`
- Added dynamic port binding using `PORT` environment variable
- Added CUDA environment configuration in startup event

### `app/models/classifier.py` & `app/models/segmentation.py`
- Added CPU-only TensorFlow configuration
- Added graceful error handling for GPU initialization

### `Dockerfile`
- Added CPU-only environment variables
- Fixed port binding in CMD
- Fixed health check to use urllib instead of requests
- Updated to use `start_render.py` startup script

### `start_render.py` (New File)
- Dedicated startup script for Render deployment
- Configures environment for CPU-only execution
- Handles port configuration properly

## Testing the Deployment

1. **Health Check**: Visit `https://your-app-name.onrender.com/health`
2. **Main Application**: Visit `https://your-app-name.onrender.com/`
3. **API Documentation**: Visit `https://your-app-name.onrender.com/docs`

## Troubleshooting

### If you still see CUDA errors:
1. Check that all environment variables are set correctly
2. Verify that the Dockerfile is using the latest version
3. Check Render logs for any startup errors

### If port binding still fails:
1. Ensure the `PORT` environment variable is set in Render
2. Check that the application is binding to `0.0.0.0` not `127.0.0.1`
3. Verify the health check endpoint is accessible

### Performance Considerations:
- CPU-only inference will be slower than GPU
- Consider using smaller model variants if needed
- Monitor memory usage in Render dashboard

## Model Files

Ensure these model files are present in your repository:
- `app/models/brain_tumor_classifier_tuned.keras`
- `app/models/final_unet_model.keras`

If models are large (>100MB), consider:
- Using Git LFS for model files
- Hosting models on cloud storage and downloading during build
- Using model quantization to reduce file sizes

## Security Notes

- The application runs as non-root user in Docker
- CORS is configured to allow all origins (consider restricting for production)
- Health checks are properly configured
- Environment variables are used for configuration

## Next Steps

After successful deployment:
1. Test the API endpoints
2. Configure custom domain if needed
3. Set up monitoring and logging
4. Consider implementing rate limiting
5. Add authentication if required for production use
