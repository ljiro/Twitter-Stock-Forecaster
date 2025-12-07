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

WORKDIR /app

# 3. Install PyTorch (CPU Only) - using extra-index-url to fix dependencies
RUN pip install --no-cache-dir "torch>=2.6" --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Install Other Deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Code
COPY . .

# 6. Run Orchestrator
CMD ["python", "src/main_orchestrator.py"]