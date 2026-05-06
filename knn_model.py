# ===========================================
#  KNN Model for Diabetes Prediction
# ===========================================

# =====================
# 📦 Imports
# =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, recall_score

# =====================
#  Load Data
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
# 🧹 Data Cleaning
# =====================
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_with_zero:
    data[col] = data[col].replace(0, data[col].median())

print("✅ Data cleaned")

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
# Pipeline + GridSearch
# =====================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="recall"   # IMPORTANT
)

print("\n🔄 Training KNN...")
grid_search.fit(X_train, y_train)

print("✅ Best Parameters:", grid_search.best_params_)
print("✅ Best CV Recall:", grid_search.best_score_)

# =====================
#  Final Model
# =====================
best_model = grid_search.best_estimator_

# =====================
#  Evaluation
# =====================
y_pred = best_model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("Recall (Diabetes):", recall_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# =====================
# ROC Curve
# =====================
y_prob = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# =====================
#  Cross-validation
# =====================
cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=StratifiedKFold(n_splits=5),
    scoring="recall"
)

print("\nCross-validation recall:", np.mean(cv_scores))

# =====================
#  Save Model
# =====================
joblib.dump(best_model, "knn_model.pkl")

print("✅ KNN model saved!")