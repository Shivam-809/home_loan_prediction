import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from train_model import train_model

# Load data
data = pd.read_csv("data/train.csv")

# Drop 'Loan_ID' if present
if "Loan_ID" in data.columns:
    data = data.drop("Loan_ID", axis=1)

# Convert Loan_Status to binary
data["Loan_Status"] = data["Loan_Status"].map({'Y': 1, 'N': 0})

# Split into features and target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Define model and hyperparameter grid
model = RandomForestClassifier(random_state=42)
param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [3, 5, 10],
    'model__min_samples_split': [2, 5]
}

# Train model
print("Training model...")
best_model = train_model(model, param_grid, X, y)

# Save the best model
joblib.dump(best_model.best_estimator_, "best_model.pkl")
print("✅ Model saved as best_model.pkl")
