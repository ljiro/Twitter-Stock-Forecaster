FROM python:3.10-slim

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    xvfb \
    wget \
    gcc \
    g++ \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 2. Environment Variables
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV DISPLAY=:99 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ADD THESE: Set Hugging Face cache to writable /tmp location
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV XDG_CACHE_HOME=/tmp

WORKDIR /app

# 3. Install PyTorch (CPU)
RUN pip install --no-cache-dir "torch>=2.6" --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Install Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Create Non-Root User
RUN useradd -m appuser

# 6. Create writable cache directory for Hugging Face in /tmp
RUN mkdir -p /tmp/huggingface && chown -R appuser:appuser /tmp/huggingface

# 7. Copy Code & Fix Permissions
COPY . .
RUN chown -R appuser:appuser /app

# 8. Switch to User
USER appuser

# 9. Run
CMD ["python", "src/main_orchestrator.py"]