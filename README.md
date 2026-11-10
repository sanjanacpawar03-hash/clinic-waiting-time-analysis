# 🏥 Clinic Waiting Time Analysis and Prediction System

## 📌 Project Overview

The Clinic Waiting Time Analysis and Prediction System is a data-driven application designed to analyze patient flow, understand waiting-time patterns, and predict clinic waiting times using Machine Learning.

The system provides interactive dashboards and predictive analytics to help clinics understand operational performance and make informed decisions.

## 🎯 Objectives

- Analyze patient flow and waiting-time patterns.
- Identify factors affecting patient waiting times.
- Predict waiting times using Machine Learning.
- Visualize clinic performance through interactive dashboards.
- Provide model explainability using SHAP.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| Database | SQLite |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Explainable AI | SHAP |

## 🤖 Machine Learning Models

- Linear Regression
- Random Forest Regressor
- LSTM (Long Short-Term Memory)

## 📊 Key Features

- Patient-flow analysis
- Waiting-time analysis
- Interactive data visualizations
- Waiting-time prediction
- Machine Learning model evaluation
- Explainable AI using SHAP
- Streamlit dashboard
- FastAPI backend

## 📁 Project Structure

```text
clinic_wait/
│
├── app.py
├── backend.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── report.pdf
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/clinic-waiting-time-analysis.git
```

### 2. Navigate to the project directory

```bash
cd clinic-waiting-time-analysis
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

### Start the Streamlit application

```bash
streamlit run app.py
```

### Start the FastAPI backend

```bash
uvicorn backend:app --reload
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## 🔒 Dataset and Privacy

This project uses simulated patient-flow data for analysis and demonstration. No real patient records are intended to be published in this repository.

## 👩‍💻 Author

**Sanjana Chandrakanth Pawar**

MCA Graduate | Aspiring Data Analyst / Data Scientist

GitHub: https://github.com/sanjanacpawar03-hash

LinkedIn: https://www.linkedin.com/in/sanjana-chandrakanth-pawar-180997371/