"""
Diabetes Prediction using Machine Learning

This script trains a classification model to predict whether a patient
has diabetes based on medical features. It includes data cleaning,
model tuning, evaluation, and comparison with other models.
"""

# =====================
# Imports
# =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE


# =====================
# Load Data
# =====================
def load_data(filepath):
    data = pd.read_csv(filepath)
    if data.empty:
        raise ValueError("Data not loaded correctly. Please check the file path.")
    return data

data = load_data('/Users/omniaabouhassan/Desktop/ML project/diabetes.csv')

print("Data preview:\n", data.head())


# =====================
# Data Cleaning
# =====================
# In this dataset, some features contain zeros that are not medically valid
# (e.g., BMI or Glucose cannot realistically be 0).
# Replace those values with the median to avoid skewing the model.

cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_with_zero:
    data[col] = data[col].replace(0, data[col].median())


# =====================
#  Quick EDA
# =====================
# Check class distribution (important for imbalanced data)

sns.countplot(x="Outcome", data=data)
plt.title("Class Distribution")
plt.show()

# Feature correlation
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# =====================
# Split Data
# =====================
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Stratified split keeps the class ratio balanced in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =====================
# Handle Imbalance
# =====================
# Apply SMOTE to balance the minority class (diabetes cases)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# =====================
# Hyperparameter Tuning (Optuna)
# =====================
# We optimize for recall because in a medical setting,
# missing a positive case (false negative) is more critical.

def objective(trial):
    C = trial.suggest_float("C", 0.01, 10, log=True)
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # scaling inside pipeline avoids data leakage
        ("model", LogisticRegression(
            C=C,
            solver=solver,
            max_iter=1000,
            class_weight="balanced"  # helps with class imbalance
        ))
    ])

    score = cross_val_score(
        pipeline,
        X_train_res,
        y_train_res,
        cv=5,
        scoring="recall"
    ).mean()

    return score


print("Running hyperparameter tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best parameters:", study.best_params)


# =====================
#  Train Final Model
# =====================
best_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        **study.best_params,
        max_iter=1000,
        class_weight="balanced"
    ))
])

best_pipeline.fit(X_train_res, y_train_res)


# =====================
# 📈 Evaluation
# =====================
y_pred = best_pipeline.predict(X_test)

print("\nModel Performance:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# ROC-AUC gives a better picture of model performance than accuracy alone
y_prob = best_pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", roc_auc)


# Confusion matrix plot
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# =====================
# Model Comparison
# =====================
# Compare with other common models used for tabular data

models = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    
    pipe.fit(X_train_res, y_train_res)
    preds = pipe.predict(X_test)
    
    print(f"\n{name} Results:")
    print(classification_report(y_test, preds))


# =====================
#  Feature Importance
# =====================
# For logistic regression, coefficients indicate feature influence

model = best_pipeline.named_steps["model"]
importance = model.coef_[0]
features = X.columns

sns.barplot(x=importance, y=features)
plt.title("Feature Importance")
plt.show()


# =====================
# Save Model
# =====================
joblib.dump(best_pipeline, "diabetes_model.pkl")

print("Model saved successfully!")