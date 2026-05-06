
"""Neural Network Models for Diabetes Prediction

This script trains and evaluates simple neural network models (MLP and DNN)
to predict diabetes using structured medical data. It focuses on comparing
deep learning performance to traditional machine learning approaches.
"""
# =====================
# Imports
# =====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    recall_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# =====================
# Load Data
# =====================
# Load and Prepare Data
def load_data(filepath):
    data = pd.read_csv(filepath)
    if data.empty:
        raise ValueError("Data not loaded correctly. Please check the file path.")
    return data

data = load_data('/Users/omniaabouhassan/Desktop/ML project/diabetes.csv')
print("Data Head:\n", data.head())
print("✅ Data Loaded")
print(data.head())

# =====================
# 🧹 Data Cleaning (IMPORTANT)
# =====================
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_with_zero:
    data[col] = data[col].replace(0, data[col].median())

print("✅ Data cleaned")

# =====================
# EDA
# =====================
sns.countplot(x="Outcome", data=data)
plt.title("Class Distribution")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# =====================
# Feature Engineering 
# =====================
data["Glucose_BMI"] = data["Glucose"] * data["BMI"]
data["Age_Insulin"] = data["Age"] * data["Insulin"]

# =====================
#  Split Data
# =====================
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================
# ⚙️ Scaling
# =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================
# Early Stopping
# =====================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# =====================
#  Model 1: MLP
# =====================
mlp = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

mlp.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n Training MLP...")
mlp.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# =====================
#  Model 2: DNN
# =====================
dnn = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

dnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n Training DNN...")
dnn.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# =====================
# Evaluation Function
# =====================
def evaluate_model(model, X_test, y_test, name):
    print(f"\n📊 Evaluating {name}")
    
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    print(classification_report(y_test, y_pred))

    print("Recall (Diabetes):", recall_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    return y_prob

# =====================
# Evaluate Models
# =====================
plt.figure(figsize=(8, 6))

y_prob_mlp = evaluate_model(mlp, X_test_scaled, y_test, "MLP")
y_prob_dnn = evaluate_model(dnn, X_test_scaled, y_test, "DNN")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# =====================
# Precision-Recall Curve
# =====================
precision_mlp, recall_mlp, _ = precision_recall_curve(y_test, y_prob_mlp)
precision_dnn, recall_dnn, _ = precision_recall_curve(y_test, y_prob_dnn)

plt.figure(figsize=(8, 6))
plt.plot(recall_mlp, precision_mlp, label="MLP")
plt.plot(recall_dnn, precision_dnn, label="DNN")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

print("\n✅ Neural network evaluation complete!")
