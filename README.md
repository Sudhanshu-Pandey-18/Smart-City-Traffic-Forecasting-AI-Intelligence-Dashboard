# Smart City Traffic Forecasting & AI Intelligence Dashboard

## Overview

This project focuses on forecasting hourly traffic volume across four major city junctions as part of a smart city initiative.  
The objective was to build a reliable forecasting system that can help manage traffic peaks and support infrastructure planning decisions.

The system predicts traffic volume based on historical patterns, time-based features, and engineered signals such as rush hour and junction density.

---

## Problem Statement

Urban traffic patterns vary significantly across working days, weekends, and special occasions.  
The goal of this project was to:

- Forecast traffic at key junctions
- Handle unpredictable traffic fluctuations
- Improve accuracy using advanced time-series modeling techniques

---

## Dataset

- Training Data: 48,120 records  
- Test Data: 11,808 records  
- Target Variable: Hourly traffic volume  
- Junctions Covered: 4 major city junctions  

---

## Feature Engineering

To capture temporal traffic behavior, the following features were engineered:

- Cyclical time encoding (sin/cos transformation of hour and day)
- Lag features
- Rolling mean features
- Rush-hour indicators
- Weekend flags
- Junction density mapping

These features helped model daily and weekly traffic patterns more effectively.

---

## Model Development

Multiple regression models were evaluated:

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

After comparison, **XGBoost Regressor** was selected as the best-performing model.

### Time Series Cross-Validation Results:
- RMSE: 3.43
- R²: 0.9186

### Final Model Performance (XGBoost):
- MAE: 1.15
- RMSE: 2.00
- R²: 0.9901
- Accuracy: 99.01%

The final model reduced forecasting error by approximately 35% compared to the Linear Regression baseline.

---

## Deployment

A Streamlit-based interactive dashboard was developed to make the system usable in real-time.

The dashboard includes:

- Live traffic volume prediction
- 24-hour forecast visualization
- Congestion detection logic
- SHAP-based model explainability for feature importance analysis

The system simulates how intelligent signal timing decisions could be supported using predictive analytics.

---

## Tech Stack

- Python
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- SHAP
- Streamlit
- Matplotlib

---

## Project Outcome

This project demonstrates:

- End-to-end ML pipeline development
- Time-series forecasting techniques
- Feature engineering for temporal data
- Model comparison and validation
- Deployment of ML models into an interactive dashboard
- Explainable AI integration

---

## Future Improvements

- Integration with real-time traffic APIs
- Deployment on cloud infrastructure
- Automated model retraining pipeline
- Scaling to additional junctions and cities

---

## Author

Sudhanshu Pandey  
B.Tech CSE | AI/ML Enthusiast
