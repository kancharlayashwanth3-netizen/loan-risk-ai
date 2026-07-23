"""
Trains a RandomForest classifier on a synthetically generated, but logically
realistic, loan-applicant dataset and saves:
  - model.pkl        -> trained sklearn Pipeline (preprocessing + classifier)
  - columns.pkl       -> the exact one-hot-encoded column order the model expects
  - metrics.json      -> quick accuracy report (for README / debugging)

Run:  python train_model.py
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N = 6000

genders = np.random.choice(['Male', 'Female'], N, p=[0.6, 0.4])
married = np.random.choice(['Yes', 'No'], N, p=[0.65, 0.35])
dependents = np.random.choice([0, 1, 2, 3], N, p=[0.5, 0.2, 0.2, 0.1])
education = np.random.choice(['Graduate', 'Not Graduate'], N, p=[0.7, 0.3])
self_employed = np.random.choice(['Yes', 'No'], N, p=[0.2, 0.8])
property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], N, p=[0.4, 0.35, 0.25])

applicant_income = np.random.gamma(shape=4.0, scale=4000, size=N).round(0)
coapplicant_income = np.where(
    married == 'Yes',
    np.random.gamma(shape=2.5, scale=2500, size=N).round(0),
    0
)
loan_amount = (np.random.gamma(shape=3.0, scale=45, size=N) * 1000).round(0)
loan_term = np.random.choice([120, 180, 240, 300, 360], N, p=[0.1, 0.15, 0.2, 0.25, 0.3])

credit_history_label = np.random.choice(['Good', 'Average', 'Poor'], N, p=[0.55, 0.3, 0.15])
credit_history_num = np.select(
    [credit_history_label == 'Good', credit_history_label == 'Average', credit_history_label == 'Poor'],
    [1, 1, 0]
)
existing_loans = np.random.choice([0, 1, 2, 3, 4], N, p=[0.4, 0.3, 0.15, 0.1, 0.05])

history_map = {'Good': 300, 'Average': 150, 'Poor': 50}
history_component = np.array([history_map[h] for h in credit_history_label])
total_income = applicant_income + coapplicant_income
cibil_score = (
    300
    + history_component
    + np.minimum(total_income / 1000 * 2, 200)
    - existing_loans * 25
    - dependents * 5
)
cibil_score = np.clip(cibil_score, 300, 900).round(0)

income_to_loan_ratio = total_income * (loan_term / 12) / (loan_amount + 1)

z = (
    -6.0
    + 8.5 * (cibil_score - 300) / 600
    + 2.5 * np.clip(income_to_loan_ratio / 5, 0, 1)
    + 1.2 * credit_history_num
    - 1.0 * (existing_loans / 4)
    - 0.4 * (dependents / 3)
)
approval_prob = 1 / (1 + np.exp(-z))
approval_prob = np.clip(approval_prob, 0.02, 0.98)
loan_status = np.where(np.random.rand(N) < approval_prob, 'Approved', 'Rejected')

df = pd.DataFrame({
    'Gender': genders,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_history_num,
    'ExistingLoans': existing_loans,
    'Property_Area': property_area,
    'CibilScore': cibil_score,
    'Loan_Status': loan_status,
})

X = df.drop(columns=['Loan_Status'])
y = (df['Loan_Status'] == 'Approved').astype(int)

X_encoded = pd.get_dummies(X, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'])
feature_columns = list(X_encoded.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")

out_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(clf, os.path.join(out_dir, 'model.pkl'))
joblib.dump(feature_columns, os.path.join(out_dir, 'columns.pkl'))
with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
    json.dump({'train_accuracy': train_acc, 'test_accuracy': test_acc, 'n_samples': N}, f, indent=2)

print("Saved model.pkl, columns.pkl, metrics.json in", out_dir)
