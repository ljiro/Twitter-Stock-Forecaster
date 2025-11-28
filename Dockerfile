FROM python:3.10-slim

# 1. Install System Deps (Chromium + Xvfb + Compilation tools)
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    xvfb \
    wget \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Environment Variables
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV DISPLAY=:99 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# 3. Install Python Deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Code
COPY . .

# 5. Run Orchestrator
CMD ["python", "src/main_orchestrator.py"]