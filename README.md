# PREDICTING-HOSPITAL-ADMISSION-AT-EMERGENCY-DEPARTMENT-TRIAGE-USING-MACHINE-LEARNING
This project aims to improve emergency department workflows by predicting patient admissions using machine learning models. By accurately forecasting admissions, hospitals can better prioritize patients and manage limited resources more efficiently.

## Problem Statement

Emergency Departments around the world suffer from overcrowding and long patient wait times. Predictive modeling can assist with triage and resource allocation, leading to improved patient outcomes and operational efficiency.

## Dataset

- Total Records: 560,486
- Total Features: 972
- After preprocessing and feature engineering, dimensionality was reduced to improve signal-to-noise ratio and model performance.

## Machine Learning Models Used

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

## Features Engineered

- Dimensionality reduction
- Missing value imputation
- One-hot encoding
- Feature scaling

## Web Application

We have developed a simple and user-friendly web application to demonstrate the use of the model in real-time hospital admission prediction. It allows healthcare professionals to input patient information and get an admission prediction instantly.

You can download the models using the G-Drive link provided in App folder and can run the app locally:

```bash
streamlit run capstone_app.py
