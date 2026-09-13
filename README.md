# Customer Churn Prediction

Predicting whether a telecom customer will churn (cancel their subscription) based on account details, contract type, and usage patterns — a real business problem companies pay a lot to solve, since retaining an existing customer is far cheaper than acquiring a new one.

## Problem Statement
Telecom companies lose significant revenue when customers churn. This project builds a machine learning model to identify customers at high risk of churning, so the business can intervene early (offers, support outreach, etc.) before losing them.

## Dataset
2,000 customer records with realistic features modeled on real-world telecom churn patterns (currency in INR):
- **Demographics:** Senior citizen status, partner, dependents
- **Account info:** Tenure (months), contract type, payment method, billing type
- **Usage/charges:** Monthly charges (₹200-3000), total charges, internet service type, tech support

Churn rate in this dataset: ~48%.

## Approach
1. **Exploratory Data Analysis (EDA)** — examined churn distribution, relationship between tenure/charges/contract type and churn.
2. **Feature Engineering** — encoded categorical variables (contract type, payment method, etc.) for modeling.
3. **Model Building** — trained and compared two models:
   - Logistic Regression (interpretable baseline)
   - Random Forest (captures non-linear patterns)
4. **Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
5. **Feature Importance** — identified which factors most strongly predict churn.

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 76.0% | 72.9% | 79.1% | 75.9% | 0.840 |
| Random Forest | 76.2% | 75.8% | 73.8% | 74.8% | 0.839 |

**Top predictors of churn:** Total charges, contract type, monthly charges, and tenure — customers on month-to-month contracts with high charges and low tenure are at the highest risk of churning.

## Key Insight (business takeaway)
Customers on **month-to-month contracts** with **short tenure** and **higher monthly bills** are the highest churn risk group. A business could use this model to proactively target these customers with retention offers or improved support before they leave.

## Files
- `churn_prediction.py` — full analysis and modeling script
- `app.py` — Streamlit web app for live predictions
- `customer_churn_data.csv` — dataset used
- `eda_overview.png` — exploratory data analysis visuals
- `model_evaluation.png` — confusion matrix and ROC curve
- `feature_importance.png` — feature importance chart
- `churn_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `feature_columns.pkl` — saved model artifacts used by the app
- `requirements.txt` — dependencies needed to run the project

## How to Run

**1. Train the model (generates the .pkl files):**
```bash
pip install -r requirements.txt
python churn_prediction.py
```

**2. Run the web app locally:**
```bash
streamlit run app.py
```
This opens an interactive browser page where you can input customer details and get a live churn prediction.

## Live Demo
Deployed on Streamlit Community Cloud: *[add your deployed link here]*

## Tools Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## What I'd Improve Next
- Try hyperparameter tuning (GridSearchCV) to improve Random Forest recall
- Test with real-world dataset (e.g. IBM Telco Customer Churn) for validation
- Add SHAP values for deeper model interpretability
- Deploy as a simple web app for interactive predictions
