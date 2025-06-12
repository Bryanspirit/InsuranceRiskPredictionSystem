import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Create sample model and scaler for demonstration
# In production, you would use your actual trained model

print("Creating sample model and scaler for demonstration...")

# Create sample data structure that matches your model
n_features = 29  # Adjust based on your actual feature count
X_sample = np.random.randn(100, n_features)
y_sample = np.random.randint(0, 2, 100)

# Create and train a sample model
model = make_pipeline(
    SimpleImputer(strategy='mean'),
    LogisticRegression(random_state=42)
)

model.fit(X_sample, y_sample)

# Create a sample scaler
scaler = StandardScaler()
scaler.fit(X_sample[:, :9])  # First 9 features are numerical

# Save the model and scaler
joblib.dump(model, "../logistic_regression_model.pkl")
joblib.dump(scaler, "../scaler.pkl")

print("Sample model and scaler created successfully!")
print("Files saved: logistic_regression_model.pkl, scaler.pkl")
print("\nNote: These are sample models for demonstration.")
print("Replace them with your actual trained models for production use.")
