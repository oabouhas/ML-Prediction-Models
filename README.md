# Diabetes Prediction Using Machine Learning

## Introduction
This project aims to predict diabetes using various machine learning classification methods. The dataset used for this project includes several health metrics and the outcome of diabetes diagnosis.

## Problem Definition
The goal is to determine the effectiveness of different classification models in predicting whether a patient has diabetes based on various health metrics.

## Dataset Overview
The dataset contains the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

## Methodology
This repository includes the following scripts:
- `logistic_regression.py`: Logistic regression for diabetes prediction.
- `Extra_models.py`: Advanced techniques and models.
- `neural_networks.py`: Various neural network architectures.

### Logistic Regression Implementation
- **Data Preprocessing**: Handling missing values and scaling features.
- **Model Training and Evaluation**: Hyperparameter tuning using Optuna and evaluating model performance.
- **Feature Importance**: Identifying key features contributing to predictions.

### Advanced Techniques
- **Extra_models.py**: Includes enhanced preprocessing, feature engineering, ensemble methods, and model interpretation.
- **neural_networks.py**: Implements and evaluates MLP, DNN, CNN, and RNN architectures.

## Results
- **Logistic Regression Accuracy**: 76%
- **Random Forest Accuracy**: 73.4%
- **SVM Accuracy**: 73.4%
- **Gradient Boosting Accuracy**: 74.0%
- **Ensemble Model Accuracy**: 77.9%
- ## Acknowledgments
- Kaggle for the dataset https://www.kaggle.com/code/rishavwalde/healthcare-data-analysis-and-predictive-modeling/input
