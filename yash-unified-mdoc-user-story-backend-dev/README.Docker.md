# Docker Setup for Meeting Document Generator

This guide explains how to build and run the Meeting Document Generator using Docker.

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 1.29 or higher) - optional but recommended

## Quick Start

### Using Docker Compose (Recommended)

1. **Create a `.env` file** in the project root with your configuration:
   ```bash
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key
   
   # Azure OpenAI Configuration (optional)
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   AZURE_OPENAI_API_KEY=your_azure_key
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   
   # Azure Speech Services (optional)
   AZURE_SPEECH_KEY=your_speech_key
   AZURE_SPEECH_REGION=your_speech_region
   
   # Storage Configuration
   LOCAL_STORAGE_DIR=data/outputs
   ```

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

4. **View logs**:
   ```bash
   docker-compose logs -f
   ```

5. **Stop the application**:
   ```bash
   docker-compose down
   ```

### Using Docker Directly

1. **Build the Docker image**:
   ```bash
   docker build -t meeting-document-generator .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name meeting-document-generator \
     -p 8501:8501 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/local_storage:/app/local_storage \
     -v $(pwd)/.env:/app/.env:ro \
     --env-file .env \
     meeting-document-generator
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

4. **View logs**:
   ```bash
   docker logs -f meeting-document-generator
   ```

5. **Stop the container**:
   ```bash
   docker stop meeting-document-generator
   docker rm meeting-document-generator
   ```

## Configuration

### Environment Variables

The application uses environment variables for configuration. Create a `.env` file in the project root with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_key_here

# Azure OpenAI (optional)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure Speech Services (optional)
AZURE_SPEECH_KEY=your_key_here
AZURE_SPEECH_REGION=your_region

# Storage
LOCAL_STORAGE_DIR=data/outputs
```

### Volume Mounts

The Docker setup mounts the following directories for persistent storage:

- `./data` → `/app/data` - Output files and generated documents
- `./local_storage` → `/app/local_storage` - Audit logs and usage cost logs
- `./config` → `/app/config` - Configuration files (optional)
- `./.env` → `/app/.env` - Environment variables (read-only)

## Building for Production

### Optimize Image Size

The Dockerfile uses a multi-stage build approach. For production, you might want to:

1. **Use a smaller base image** (if you don't need all features):
   ```dockerfile
   FROM python:3.11-slim
   ```

2. **Remove development dependencies** from requirements.txt

3. **Use .dockerignore** to exclude unnecessary files

### Build Arguments

You can customize the build with build arguments:

```bash
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  -t meeting-document-generator:latest .
```

## Troubleshooting

### Container won't start

1. **Check logs**:
   ```bash
   docker-compose logs meeting-document-generator
   ```

2. **Verify environment variables**:
   ```bash
   docker-compose exec meeting-document-generator env
   ```

3. **Check port availability**:
   ```bash
   # Check if port 8501 is already in use
   lsof -i :8501
   ```

### FFmpeg not found

FFmpeg is installed in the Docker image. If you see errors:

1. **Verify FFmpeg installation**:
   ```bash
   docker-compose exec meeting-document-generator ffmpeg -version
   ```

2. **Check if the binary is in PATH**:
   ```bash
   docker-compose exec meeting-document-generator which ffmpeg
   ```

### Tesseract OCR errors

Tesseract is installed in the Docker image. Verify installation:

```bash
docker-compose exec meeting-document-generator tesseract --version
```

### Memory Issues

If you encounter memory issues with large videos:

1. **Increase Docker memory limit** in Docker Desktop settings
2. **Adjust resource limits** in `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 16G  # Increase as needed
   ```

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Fix permissions (Linux/Mac)
sudo chown -R $USER:$USER data local_storage
```

## Development with Docker

### Mount source code for development

For development, you can mount the source code as a volume:

```yaml
volumes:
  - .:/app
  - /app/__pycache__  # Exclude cache
```

### Hot reload

Streamlit supports hot reload. Changes to Python files will automatically reload the app.

## Production Deployment

### Using Docker Compose

1. **Set production environment variables**:
   ```bash
   export ENV=production
   ```

2. **Use production compose file** (if you create one):
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Using Kubernetes

You can deploy this container to Kubernetes. Example deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: meeting-document-generator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: meeting-document-generator
  template:
    metadata:
      labels:
        app: meeting-document-generator
    spec:
      containers:
      - name: app
        image: meeting-document-generator:latest
        ports:
        - containerPort: 8501
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: openai-api-key
```

## Security Considerations

1. **Never commit `.env` files** to version control
2. **Use secrets management** in production (Docker secrets, Kubernetes secrets, etc.)
3. **Limit container resources** to prevent resource exhaustion
4. **Use read-only mounts** for configuration files where possible
5. **Keep base images updated** for security patches

## Support

For issues related to Docker setup, check:
- Docker logs: `docker-compose logs`
- Container status: `docker-compose ps`
- Application health: `http://localhost:8501/_stcore/health`

