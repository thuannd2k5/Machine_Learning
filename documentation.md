# BIKE RENTAL DEMAND PREDICTION USING MACHINE LEARNING

---

# 1. PROJECT OVERVIEW

## Project Title
Bike Rental Demand Prediction Using Machine Learning

---

## Problem Statement

Bike sharing systems are becoming increasingly popular in modern cities because they provide a convenient and environmentally friendly transportation solution. However, bike rental companies often face challenges in predicting customer demand at different times of the day.

If the company cannot estimate rental demand accurately:
- Some stations may run out of bikes during peak hours.
- Some locations may have too many unused bikes.
- Staff and maintenance resources may not be allocated efficiently.

Therefore, this project aims to build a Machine Learning model that can predict the number of bikes rented per hour based on environmental and time-related conditions.

---

## Business Objective

The prediction system can help bike rental companies:
- Predict peak rental hours.
- Optimize bike distribution.
- Improve operational planning.
- Manage maintenance schedules.
- Improve customer satisfaction.

---

# 2. DATASET INFORMATION

## Dataset Name
Bike Sharing Dataset

## Dataset Source
UCI Machine Learning Repository

## Dataset Description
The dataset contains hourly bike rental records collected from the Capital Bikeshare system in Washington D.C., USA during 2011 and 2012.

The dataset includes:
- Time information
- Weather conditions
- Temperature
- Humidity
- Wind speed
- Number of rented bikes

---

## Dataset Used
This project uses:

```text
hour.csv
```

because it provides detailed hourly rental information, which is more suitable for demand forecasting and business analysis.

---

## Dataset Shape
- Rows: 17,379
- Columns: 17

---

# 3. PROJECT WORKFLOW

```text
Data Understanding
→ Data Cleaning
→ Exploratory Data Analysis (EDA)
→ Business Insight Extraction
→ Feature Engineering
→ Machine Learning Modeling
→ Model Evaluation
→ GUI Development
→ Final Demo
```

---

# 4. PROJECT STRUCTURE

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

## Folder Description

### data/
Contains the dataset files.

### notebooks/
Used for:
- Exploratory Data Analysis (EDA)
- Visualization
- Model experiments

### src/
Contains the main source code:
- preprocessing
- training
- prediction

### models/
Stores trained Machine Learning models.

Example:
```text
random_forest.pkl
```

### app/
Contains the Streamlit GUI application.

### images/
Stores charts and figures for reports and presentations.

---

# 5. FEATURE DESCRIPTION

| Column | Description |
|---|---|
| season | Season of the year |
| yr | Year |
| mnth | Month |
| hr | Hour |
| holiday | Whether the day is a holiday |
| weekday | Day of the week |
| workingday | Working day or weekend |
| weathersit | Weather condition |
| temp | Temperature |
| atemp | Feeling temperature |
| hum | Humidity |
| windspeed | Wind speed |
| cnt | Total bike rentals |

---

# 6. TARGET VARIABLE

## Target Column
```text
cnt
```

This is the total number of rented bikes per hour.

The Machine Learning model will predict this value.

---

# 7. FEATURE SELECTION

## Features To Keep

| Features |
|---|
| season |
| yr |
| mnth |
| hr |
| holiday |
| weekday |
| workingday |
| weathersit |
| temp |
| atemp |
| hum |
| windspeed |

---

## Features To Remove

| Column | Reason |
|---|---|
| instant | Only an index ID |
| dteday | Duplicate time information |
| casual | Data leakage |
| registered | Data leakage |

---

## Why Remove casual and registered?

Because:

```text
cnt = casual + registered
```

If these columns are used as input features, the model will indirectly know the answer before prediction, causing data leakage.

---

# 8. DATA CLEANING & PREPROCESSING

## 8.1 Load Dataset

```python
import pandas as pd

df = pd.read_csv("../data/hour.csv")
```

---

## 8.2 Dataset Overview

```python
df.head()
df.info()
df.shape
```

Purpose:
- Understand dataset structure
- Check data types
- Determine number of rows and columns

---

## 8.3 Missing Values

```python
df.isnull().sum()
```

Purpose:
- Detect missing values
- Ensure dataset completeness

---

## 8.4 Duplicate Values

```python
df.duplicated().sum()
```

Purpose:
- Detect duplicated records
- Prevent duplicated training samples

---

## 8.5 Unique Values

```python
df["season"].unique()
```

Purpose:
- Check categorical consistency
- Verify encoded values

---

## 8.6 Statistical Description

```python
df.describe()
```

Purpose:
- Understand:
  - mean
  - median
  - standard deviation
  - minimum
  - maximum
  - quartiles

---

# 9. OUTLIER DETECTION

## IQR Method

The project uses the IQR (Interquartile Range) method to detect outliers.

genui{"math_block_widget_always_prefetch_v2":{"content":"IQR = Q3 - Q1"}}

Outlier conditions:

genui{"math_block_widget_always_prefetch_v2":{"content":"x < Q1 - 1.5(IQR) \\quad \\text{or} \\quad x > Q3 + 1.5(IQR)"}}

---

## Important Note

Some high rental counts may represent real peak-hour demand instead of noise.

Therefore, outlier removal must be considered carefully to avoid removing important business patterns.

---

# 10. EXPLORATORY DATA ANALYSIS (EDA)

## Objective of EDA

The purpose of EDA is not only to view numbers, but also to discover patterns and relationships inside the dataset.

Main questions:
- Which hours have the highest rental demand?
- How does weather affect bike rentals?
- Do working days differ from weekends?
- Does temperature influence demand?

---

# 11. UNIVARIATE ANALYSIS

## Analyze Single Variables

Examples:
- Distribution of cnt
- Distribution of hr
- Distribution of temperature

### Recommended Charts
- Histogram
- Boxplot
- Countplot

---

# 12. BIVARIATE ANALYSIS

## Analyze Relationships Between Two Variables

### 1. Hour vs Bike Rentals

Question:
```text
When are the peak rental hours?
```

Visualization:
- Lineplot
- Barplot

Expected Insight:
- Morning peak: 7AM–9AM
- Evening peak: 5PM–7PM

---

### 2. Weather vs Bike Rentals

Question:
```text
Does bad weather reduce bike rentals?
```

Visualization:
- Barplot
- Boxplot

Expected Insight:
- Clear weather increases rentals
- Rainy weather decreases rentals

---

### 3. Temperature vs Bike Rentals

Question:
```text
Does comfortable temperature increase rentals?
```

Visualization:
- Scatterplot
- Regression plot

---

### 4. Working Day vs Rentals

Question:
```text
Do working days have higher demand?
```

Visualization:
- Barplot

---

### 5. Season vs Rentals

Question:
```text
Which season has the highest bike demand?
```

Visualization:
- Barplot

---

# 13. MULTIVARIATE ANALYSIS

## Analyze Multiple Variables Together

Examples:
- hr + weathersit + cnt
- season + hr + cnt
- workingday + hr + cnt

Purpose:
- Discover more complex patterns
- Understand combined effects between variables

---

# 14. CORRELATION ANALYSIS

## Correlation Heatmap

Purpose:
- Measure relationships between features
- Detect highly correlated variables
- Support feature selection

Example:
- temp and atemp may be highly correlated

---

# 15. BUSINESS INSIGHTS

Examples of expected business insights:

- Peak rental demand occurs during commuting hours.
- Clear weather significantly increases bike rentals.
- Rain and snow reduce customer demand.
- Working days have different rental patterns compared to weekends.
- Moderate temperatures encourage bike usage.

These insights help bike rental companies improve operational planning and bike allocation.

---

# 16. MACHINE LEARNING MODELING

## Problem Type

```text
Regression
```

The goal is to predict:

```text
cnt
```

---

# 17. TRAIN-TEST SPLIT

```python
from sklearn.model_selection import train_test_split

X = df.drop("cnt", axis=1)
y = df["cnt"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 18. MACHINE LEARNING MODELS

## 1. Linear Regression

Purpose:
- Baseline model
- Easy to interpret
- Fast training

```python
from sklearn.linear_model import LinearRegression
```

---

## 2. Random Forest Regressor

Purpose:
- Handles nonlinear relationships well
- Works effectively on tabular data
- Usually achieves higher accuracy

```python
from sklearn.ensemble import RandomForestRegressor
```

---

## 3. XGBoost (Optional)

Purpose:
- Advanced boosting algorithm
- Can improve prediction performance

---

# 19. MODEL EVALUATION

## Evaluation Metrics

### MAE
Mean Absolute Error

### RMSE
Root Mean Squared Error

### R² Score
Measures prediction accuracy

---

## Example

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

---

## Model Comparison Table

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ... | ... | ... |
| Random Forest | ... | ... | ... |
| XGBoost | ... | ... | ... |

---

# 20. FEATURE IMPORTANCE

After training the Random Forest model, feature importance analysis will be performed.

Purpose:
- Identify the most influential features.
- Understand which variables affect bike demand the most.

Expected important features:
- hr
- temp
- workingday
- weathersit

---

# 21. MODEL SAVING

```python
import joblib

joblib.dump(model, "../models/random_forest.pkl")
```

---

# 22. GUI DEVELOPMENT

## Framework

```text
Streamlit
```

---

## GUI Inputs

Users can enter:
- Hour
- Season
- Weather condition
- Temperature
- Humidity
- Wind speed

---

## GUI Output

```text
Predicted Bike Rentals = XXX
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```

---

# 23. EXPECTED FINAL OUTPUT

The final system should:
- Analyze bike rental behavior.
- Discover business insights.
- Predict bike rental demand accurately.
- Provide a user-friendly prediction interface.

---

# 24. CONCLUSION

This project combines:
- Data Analysis
- Business Intelligence
- Machine Learning
- Visualization
- Real-world demand forecasting

The system can help bike rental companies make better operational decisions and improve service efficiency.

