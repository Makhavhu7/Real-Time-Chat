# Dockerfile for Real-Chat-App
# Build: docker build -t real-chat-app .
# Run: docker run --rm -i real-chat-app

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY assistant.py .
COPY config.json .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import ollama; ollama.list()" || exit 1

# Run the assistant
CMD ["python", "assistant.py"]
