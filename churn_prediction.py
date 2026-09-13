"""
Customer Churn Prediction Project
-----------------------------------
Goal: Predict whether a telecom customer will churn (leave the service)
based on their account information and usage patterns.

Author: [Your Name]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

np.random.seed(42)
sns.set_style("whitegrid")

# -----------------------------------------------------------------------
# 1. GENERATE A REALISTIC CUSTOMER CHURN DATASET
# -----------------------------------------------------------------------
# NOTE: This is a synthetically generated dataset built to realistically
# mimic real-world telecom churn data (like the well-known IBM Telco
# Churn dataset), since it lets us control and explain exactly why
# customers churn — useful for explaining the project in interviews.

n = 2000

tenure = np.random.exponential(scale=24, size=n).clip(0, 72).astype(int)
monthly_charges = np.random.normal(65, 25, n).clip(18, 120)
contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
)
internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.45, 0.20]
)
tech_support = np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
paperless_billing = np.random.choice(["Yes", "No"], size=n, p=[0.6, 0.4])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    size=n, p=[0.35, 0.2, 0.225, 0.225]
)
senior_citizen = np.random.choice([0, 1], size=n, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
total_charges = (monthly_charges * tenure) + np.random.normal(0, 50, n)
total_charges = total_charges.clip(0, None)

# Build churn probability from realistic business logic, then sample labels
churn_logit = (
    -1.5
    + 1.8 * (contract == "Month-to-month")
    - 1.2 * (contract == "Two year")
    + 0.02 * (monthly_charges - 65)
    - 0.05 * (tenure - 24)
    + 0.8 * (internet_service == "Fiber optic")
    - 0.6 * (tech_support == "Yes")
    + 0.5 * (payment_method == "Electronic check")
    + 0.3 * (paperless_billing == "Yes")
    - 0.3 * (partner == "Yes")
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churn = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    "tenure": tenure,
    "MonthlyCharges": monthly_charges.round(2),
    "TotalCharges": total_charges.round(2),
    "Contract": contract,
    "InternetService": internet_service,
    "TechSupport": tech_support,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Churn": churn
})

df.to_csv("/home/claude/customer_churn_data.csv", index=False)
print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Churn rate: {df['Churn'].mean():.2%}")

# -----------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------
print("\n--- Basic Info ---")
print(df.describe(include="all").T)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.countplot(data=df, x="Churn", ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Churn Distribution (0 = Stayed, 1 = Churned)")

sns.boxplot(data=df, x="Churn", y="tenure", ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Tenure vs Churn")

sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title("Churn by Contract Type")
axes[1, 0].tick_params(axis='x', rotation=15)

sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title("Monthly Charges vs Churn")

plt.tight_layout()
plt.savefig("/home/claude/eda_overview.png", dpi=150)
plt.close()
print("\nSaved: eda_overview.png")

# -----------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------------------------------------------
model_df = df.copy()
categorical_cols = [
    "Contract", "InternetService", "TechSupport", "PaperlessBilling",
    "PaymentMethod", "Partner", "Dependents"
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    label_encoders[col] = le

X = model_df.drop(columns=["Churn"])
y = model_df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------
# 4. MODEL TRAINING - COMPARE TWO MODELS
# -----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs),
        "preds": preds,
        "probs": probs
    }

print("\n--- Model Comparison ---")
comparison_df = pd.DataFrame({
    name: {k: v for k, v in res.items() if k not in ["model", "preds", "probs"]}
    for name, res in results.items()
}).T
print(comparison_df.round(3))

# -----------------------------------------------------------------------
# 5. CONFUSION MATRIX + ROC CURVE (best model: Random Forest)
# -----------------------------------------------------------------------
best_model_name = "Random Forest"
best = results[best_model_name]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_test, best["preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
axes[0].set_title(f"Confusion Matrix ({best_model_name})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

fpr, tpr, _ = roc_curve(y_test, best["probs"])
axes[1].plot(fpr, tpr, label=f"AUC = {best['auc']:.3f}", color="darkorange")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

plt.tight_layout()
plt.savefig("/home/claude/model_evaluation.png", dpi=150)
plt.close()
print("\nSaved: model_evaluation.png")

# -----------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (Random Forest)
# -----------------------------------------------------------------------
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("/home/claude/feature_importance.png", dpi=150)
plt.close()
print("\nSaved: feature_importance.png")

print("\n--- Top 5 Most Important Features for Predicting Churn ---")
print(importances.head(5))

print("\n--- Classification Report (Random Forest) ---")
print(classification_report(y_test, best["preds"], target_names=["Stayed", "Churned"]))

print("\nProject run completed successfully.")
