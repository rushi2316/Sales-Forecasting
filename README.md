# 📈 Sales Forecasting Dashboard

> An interactive machine learning-based sales forecasting application built with Python and Streamlit to analyze historical sales data, compare multiple regression models, and generate sales predictions for selected stores and product families.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)

---

## 🚀 Live Demo

### 🌐 Try the Application

**Live Streamlit App:**

PASTE_YOUR_STREAMLIT_URL_HERE

### 💻 Source Code

**GitHub Repository:**

PASTE_YOUR_GITHUB_REPOSITORY_URL_HERE

---

## 📌 Project Overview

Sales forecasting helps businesses estimate future demand and make better decisions related to inventory, operations, and planning.

This project develops an interactive **Sales Forecasting Dashboard** that uses historical store-level sales data and machine learning techniques to predict sales.

The application allows users to:

- Select a specific store
- Select a product family
- Analyze historical sales data
- Train multiple machine learning models
- Compare model performance
- Evaluate models using MAE and RMSE
- Visualize actual vs predicted sales
- Analyze feature importance
- Explore feature correlations
- Generate a 30-day forecast using Prophet
- Enter custom sales values
- Generate sales predictions
- Download prediction results as CSV

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze historical sales patterns.
2. Perform data preprocessing and cleaning.
3. Create meaningful time-series features.
4. Build multiple machine learning regression models.
5. Compare model performance using evaluation metrics.
6. Combine predictions using ensemble learning.
7. Build an interactive forecasting dashboard.
8. Deploy the application as a web-based data science application.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and evaluation |
| XGBoost | Gradient boosting regression |
| Prophet | Time-series forecasting |
| Streamlit | Interactive web application |
| Git | Version control |
| GitHub | Source code hosting |
| Streamlit Community Cloud | Application deployment |

---

## 📊 Dataset

The project uses the **Store Sales - Time Series Forecasting** dataset.

The dataset contains historical sales information for multiple stores and product families.

### Important Columns

| Column | Description |
|---|---|
| `id` | Unique record identifier |
| `date` | Sales date |
| `store_nbr` | Store number |
| `family` | Product family |
| `sales` | Sales value |
| `onpromotion` | Number of products on promotion |

The application dynamically filters the data based on the selected:

**Store + Product Family**

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1. Date Conversion

The `date` column is converted into Pandas datetime format.

### 2. Sorting

The dataset is sorted chronologically to preserve the time-series structure.

### 3. Store and Product Filtering

Users can select a specific store and product family from the Streamlit sidebar.

### 4. Missing Values

Rows containing unavailable lag and rolling features are removed before model training.

---

## ⚙️ Feature Engineering

The project creates several time-series features from historical sales.

### 📅 Day of Week

```text
dayofweek