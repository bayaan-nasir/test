# Use official Python slim image
FROM python:3.10-slim

# Set the working directory to the root of the project
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the inner folder
COPY ml_service/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire root directory into the container
COPY . .

# Ensure the media directory exists for Grad-CAM heatmaps
RUN mkdir -p media/gradcam

# Set PYTHONPATH so Python recognizes ml_service as a module
ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "8001"]