
"""
Advanced Diabetes Prediction Pipeline script builds a complete machine learning pipeline for diabetes prediction.Advanced Diabetes Prediction Pipeline
It includes data preprocessing, feature engineering, model comparison,
hyperparameter tuning, and an optional prediction interface.
"""


# =====================
#  Imports
# =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

RANDOM_STATE = 42


# =====================
# Load Data
# =====================
def load_data(filepath):
    """Load dataset and handle invalid zero values."""
    
    data = pd.read_csv(filepath)

    print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")

    # Replace medically impossible zeros with NaN
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in cols:
        data.loc[data[col] == 0, col] = np.nan

    return data


# ✅ ✅ CALL FUNCTION CORRECTLY HERE (OUTSIDE)
data = load_data('/Users/omniaabouhassan/Desktop/ML project/diabetes.csv')


# =====================
# Feature Engineering
# =====================
def engineer_features(data):
    df = data.copy()

    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Age_Insulin'] = df['Age'] * df['Insulin']

    df['Log_Glucose'] = np.log1p(df['Glucose'])
    df['Log_BMI'] = np.log1p(df['BMI'])

    return df


# =====================
# Visualization
# =====================
def visualize_data(data):

    sns.countplot(x='Outcome', data=data)
    plt.title('Outcome Distribution')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


# =====================
#  Data Prep
# =====================
def prepare_data(data):

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
    )

    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test, imputer, scaler


# =====================
# Train Models
# =====================
def train_models(X_train, y_train, X_test, y_test):

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'SVM': SVC(probability=True)
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = f1_score(y_test, y_pred)

        print(classification_report(y_test, y_pred))

        if score > best_score:
            best_score = score
            best_model = model

    return best_model


# =====================
# Optimize Model
# =====================
def optimize_best_model(model, X_train, y_train):

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }

    search = RandomizedSearchCV(model, param_grid, n_iter=5, scoring='f1')

    search.fit(X_train, y_train)

    return search.best_estimator_


# =====================
#  Evaluation
# =====================
def evaluate_final_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.title("Confusion Matrix")
    plt.show()


# =====================
# Save Model
# =====================
def save_model(model, imputer, scaler):

    joblib.dump({
        "model": model,
        "imputer": imputer,
        "scaler": scaler
    }, "diabetes_model.pkl")

    print("Model saved!")


# =====================
# MAIN PIPELINE
# =====================
def main():

    visualize_data(data)

    df = engineer_features(data)

    X_train, X_test, y_train, y_test, imputer, scaler = prepare_data(df)

    model = train_models(X_train, y_train, X_test, y_test)

    model = optimize_best_model(model, X_train, y_train)

    evaluate_final_model(model, X_test, y_test)

    save_model(model, imputer, scaler)


if __name__ == "__main__":
    main()