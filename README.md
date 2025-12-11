# NVIDIA Forecasting Pipeline

## Project Overview

This project is a research-grade Machine Learning pipeline designed to predict the price **returns** of NVIDIA (NVDA) stock over short-term horizons (1-day, 2-day, 3-day). Unlike standard regression which predicts a single point (the mean), this system uses **Quantile Regression** to forecast a **Cone of Uncertainty** (risk bounds), providing a statistically rigorous measure of price volatility.

The core value lies in integrating **alternative data** (Social Sentiment and Market Fear) with robust deep learning and tree-based models, validated using rigorous anti-leakage techniques.

## 🧱 Architecture and Data Flow

The system runs on a modern, decoupled 3-tier architecture, mimicking a production ML microservice environment.

1.  **Orchestrator (Python/APScheduler):** Manages the entire ML lifecycle.
2.  **API (FastAPI):** Acts as the data bridge, serving live JSON predictions and metrics.
3.  **UI (Streamlit):** Visualizes the full analytical report and model comparisons.

### **Data Flow:**
$$
\text{Raw Data (Tweets/API)} \xrightarrow[\text{DB Update}]{\text{Orchestrator}} \text{SQLite Database} \xrightarrow[\text{Training/Inference}]{\text{Orchestrator}} \text{JSON/CSV Files} \xrightarrow[\text{API Endpoints}]{\text{FastAPI}} \text{Streamlit UI}
$$

## ✨ Key Features & Research Focus

This pipeline is built specifically to address the challenges of financial time-series forecasting:

* **Multi-Horizon Forecasting (Path Prediction):** Trains separate models for 1-Day, 2-Day, and 3-Day targets. The UI visualizes the resulting **trajectory curve** rather than a single jump to the 3-Day target.
* **Quantile Regression (Risk Modeling):** Uses the **Pinball Loss** function to predict 10th and 90th percentile bounds, creating the **80% Confidence Interval** (the "Cone of Uncertainty").
* **Purged Validation:** Employs a time-series *purging gap* in the training loop to eliminate the look-ahead bias common in financial feature engineering (e.g., RSI calculation).
* **Winkler Score Tracking:** Models are benchmarked using the Winkler Score—a professional metric that simultaneously rewards high accuracy and punishes intervals that are too wide or too narrow.
* **Comparative Ensemble:** Compares the performance of four different algorithms: Gradient Boosting (GBR), Linear Quantile Regression, Multi-Quantile LSTM (MQLSTM), and Quantile Regression Neural Network (QRNN).

## 💻 Technical Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | Python, `apscheduler` | Schedules training and inference loops. |
| **Data/ML** | `pandas`, `numpy`, `scikit-learn` (GBR), `TensorFlow/Keras` (LSTM/QRNN) | Model training, feature engineering, and rigorous validation. |
| **Data Storage** | `SQLite` (`nvda.db`), JSON, CSV | Handles raw data and serves real-time inference payloads. |
| **Backend** | `FastAPI`, `uvicorn` | Lightweight API serving predictions (`/predict`) and metrics (`/performance`). |
| **Frontend** | `Streamlit` | Interactive report generation and visualization of the "Battle Arena." |

## 🚀 Getting Started

Follow these steps to clone the repository and launch the full system locally.

### Prerequisites

You must have Python 3.10+ installed.

```bash
# 1. Clone the repository
git clone https://github.com/ljiro/Twitter-Stock-Forecaster.git
cd Twitter-Stock-Forecaster

# 2. Install dependencies
pip install -r requirements.txt 
# Note: Ensure you have all necessary packages installed (e.g., pandas, numpy, tensorflow, fastapi, uvicorn, streamlit, apscheduler, scikit-learn).

```
# Installation Steps

The pipeline requires three concurrent processes to run continuously.

#Step 1: Start the Orchestrator (Data/ML Backend)
The orchestrator builds the database, trains the models, and saves the prediction files (data/latest_prediction.json, data/live_predictions.csv).


```bash

# Terminal 1: Run the main training/inference loop
python main_orchestrator.py
# Wait for the log messages: 
# ✅ Models Retrained...
# ✅ Inference Complete...
Step 2: Start the API Server
The API reads the JSON/CSV files created in Step 1 and makes them available to the frontend.

```
# Step2: Run FastAPI server
```bash

# Terminal 2: Run the FastAPI server
uvicorn src/api:app --reload
# Console should show: Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000)
Step 3: Launch the User Interface
The Streamlit app connects to the API and renders the interactive report.
```
# Step3: Run run Streamlit App
```bash

# Terminal 3: Run the Streamlit app
streamlit run src/app.py
# The app will open in your browser, displaying the live analysis.
```

# 📂 Project Structure
```bash

Twitter-Stock-Forecaster/
├── main_orchestrator.py      <-- (The master scheduler/trainer)
├── src/
│   ├── api.py                  <-- (FastAPI endpoints: /predict, /history)
│   ├── app.py                  <-- (Streamlit frontend UI)
│   ├── analysis/
│   │   ├── model.py            <-- (NvidiaQuantileModel base class)
│   │   └── adapted_rivals.py     <-- (MQLSTM, QRNN, Linear QR classes)
│   ├── data_collection/        <-- (Scraping, institutional data helpers)
│   ├── data_preprocessing/     <-- (Sentiment, feature engineering)
│   └── features.py             <-- (Technical indicator calculation)
└── data/
    ├── nvda.db                 <-- (SQLite Database - Raw data history)
    ├── latest_prediction.json  <-- (Real-time payload for API)
    └── live_predictions.csv    <-- (History for graph context)

```
# Application preview 
<img width="1920" height="911" alt="image" src="https://github.com/user-attachments/assets/4dcc98d0-27cf-4521-9918-8c603e72dda0" />
<img width="1918" height="912" alt="image" src="https://github.com/user-attachments/assets/028a0514-c807-44ae-9127-bf97ba80f7ad" />

