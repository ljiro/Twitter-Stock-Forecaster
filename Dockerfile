# Base Image: Lightweight Python 3.10 on Debian
FROM python:3.10-slim

# -----------------------------------------------------------------------------
# 1. SYSTEM DEPENDENCIES
# -----------------------------------------------------------------------------
# chromium & chromium-driver: The browser engine for Scweet
# xvfb: X Virtual Framebuffer (Fake monitor for headless=False)
# gcc/g++: Compilers needed for some Python math libraries
# procps: Adds 'ps' command, sometimes needed by Selenium to kill processes
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    xvfb \
    wget \
    gcc \
    g++ \
    procps \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
# Point Selenium to the installed Chromium binaries
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Configure Xvfb Display (The "Ghost Monitor")
ENV DISPLAY=:99 

# Python Performance Tweaks
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# -----------------------------------------------------------------------------
# 3. PYTHON SETUP
# -----------------------------------------------------------------------------
WORKDIR /app

# A. Install PyTorch CPU-only FIRST
# We do this separately because it's huge (~800MB). 
# Installing it here lets Docker cache this layer so you don't re-download it 
# every time you change your requirements.txt.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# B. Install remaining dependencies
COPY requirements.txt .
# We exclude torch from requirements.txt (or let pip skip it if present) to avoid downloading the GPU version
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# 4. APPLICATION CODE
# -----------------------------------------------------------------------------
COPY . .

# -----------------------------------------------------------------------------
# 5. EXECUTION
# -----------------------------------------------------------------------------
# Run the orchestrator script
CMD ["python", "src/main_orchestrator.py"]