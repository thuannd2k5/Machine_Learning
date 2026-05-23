# Bike Rental Demand Prediction 🚲

Machine Learning project for predicting hourly bike rental demand based on weather, time, and environmental conditions.

---

# 📌 Project Overview

This project uses the **Bike Sharing Dataset** to analyze and predict the number of bikes rented per hour using Machine Learning algorithms.

The system helps bike rental companies:
- Predict peak rental hours
- Optimize bike distribution
- Improve operational planning
- Support maintenance scheduling

---

# 🎯 Objective

Build a Machine Learning model that predicts:

```text
cnt = total number of rented bikes per hour
```

based on:
- Hour
- Weather
- Temperature
- Humidity
- Season
- Working day
- Wind speed
- Other environmental features

---

# 📂 Dataset

Dataset used:

```text
hour.csv
```

Dataset information:
- Source: UCI Machine Learning Repository
- Rows: 17,379
- Columns: 17
- Type: Hourly bike rental records

---

# 🧠 Machine Learning Task

```text
Regression
```

The goal is to predict hourly bike rental demand.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

# 📁 Project Structure

```text
bike-sharing-ml/
│
├── data/
│   └── hour.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   └── random_forest.pkl
│
├── app/
│   └── app.py
│
├── images/
│
├── requirements.txt
│
└── README.md
```

---

# 🔍 Main Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Business Insight Extraction
- Hourly Demand Prediction
- Model Comparison
- Streamlit GUI Application

---

# 📊 Exploratory Data Analysis

The project analyzes:
- Peak rental hours
- Weather impact on rentals
- Working day vs weekend behavior
- Seasonal trends
- Temperature influence

Visualization examples:
- Hour vs Bike Rentals
- Weather vs Rentals
- Correlation Heatmap
- Temperature vs Rentals

---

# 🤖 Models Used

## 1. Linear Regression
Baseline regression model.

## 2. Random Forest Regressor
Main prediction model for handling nonlinear relationships.

## 3. XGBoost (Optional)
Advanced boosting algorithm for performance improvement.

---

# 📈 Evaluation Metrics

Models are evaluated using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

# ⚠️ Feature Selection

## Features Removed

| Column | Reason |
|---|---|
| instant | Index column |
| dteday | Duplicate time information |
| casual | Data leakage |
| registered | Data leakage |

---

# ▶️ Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

---

# 🚀 Run Project

## Run Jupyter Notebook

```bash
jupyter notebook
```

---

## Run Streamlit GUI

```bash
streamlit run app/app.py
```

---

# 🖥 GUI Features

Users can input:
- Hour
- Weather condition
- Temperature
- Humidity
- Wind speed

The system predicts:

```text
Predicted Bike Rentals = XXX
```

---

# 📌 Expected Business Insights

- Peak rental demand occurs during commuting hours.
- Bad weather reduces bike rentals.
- Moderate temperatures increase demand.
- Working days have different patterns compared to weekends.

---

# 👨‍💻 Future Improvements

- Add XGBoost optimization
- Deploy application online
- Add real-time weather API
- Create interactive dashboard

---

# 📚 Dataset Source

Bike Sharing Dataset  
UCI Machine Learning Repository

---

# 📄 License

This project is for educational and research purposes.
