# Base image - using slim to keep the Hugging Face container size small
FROM python:3.11-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Working directory
WORKDIR /app

# Install system dependencies (required for OpenCV and ChromaDB)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files and the 200MB agrifix_db
COPY . .

# Create runtime directories
RUN mkdir -p cache temp_uploads

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--loop", "uvloop"]