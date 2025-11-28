FROM python:3.10-slim

# Install Chromium, Driver, AND Xvfb (Virtual Monitor)
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    xvfb \               
    wget \
    && rm -rf /var/lib/apt/lists/*

# Point Python to the installed Chromium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV DISPLAY=:99 

WORKDIR /app

# Ensure pyvirtualdisplay is in your requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# (CMD depends on your orchestrator, e.g., python src/main.py)