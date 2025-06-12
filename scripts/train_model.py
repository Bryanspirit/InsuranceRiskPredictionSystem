import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

print("Starting model training process...")

# For demonstration, we'll create sample data that matches your structure
# In production, replace this with your actual CSV file loading
print("Creating sample data (replace with your actual CSV loading)...")

# Sample data creation (replace this section with your actual data loading)
np.random.seed(42)
n_samples = 1000

# Create sample data that matches your AutoInsurance.csv structure
sample_data = {
    'Customer': [f'Customer_{i}' for i in range(n_samples)],
    'State': np.random.choice(['CA', 'NY', 'TX', 'FL'], n_samples),
    'Customer Lifetime Value': np.random.normal(8000, 2000, n_samples),
    'Response': np.random.choice(['No', 'Yes'], n_samples),
    'Coverage': np.random.choice(['Basic', 'Extended', 'Premium'], n_samples),
    'Education': np.random.choice(['High School', 'Bachelor', 'College', 'Master', 'Doctor'], n_samples),
    'Effective To Date': ['2023-12-31'] * n_samples,
    'EmploymentStatus': np.random.choice(['Employed', 'Unemployed', 'Retired'], n_samples),
    'Gender': np.random.choice(['M', 'F'], n_samples),
    'Income': np.random.normal(50000, 15000, n_samples),
    'Location Code': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
    'Marital Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
    'Monthly Premium Auto': np.random.normal(100, 30, n_samples),
    'Months Since Last Claim': np.random.randint(0, 60, n_samples),
    'Months Since Policy Inception': np.random.randint(1, 120, n_samples),
    'Number of Open Complaints': np.random.randint(0, 5, n_samples),
    'Number of Policies': np.random.randint(1, 10, n_samples),
    'Policy Type': np.random.choice(['Personal Auto', 'Corporate Auto'], n_samples),
    'Policy': [f'Policy_{i}' for i in range(n_samples)],
    'Renew Offer Type': np.random.choice(['Offer1', 'Offer2', 'Offer3'], n_samples),
    'Sales Channel': np.random.choice(['Agent', 'Branch', 'Call Center', 'Web'], n_samples),
    'Total Claim Amount': np.random.exponential(500, n_samples),
    'Vehicle Class': np.random.choice(['Two-Door Car', 'Four-Door Car', 'SUV', 'Luxury Car'], n_samples),
    'Vehicle Size': np.random.choice(['Small', 'Medsize', 'Large'], n_samples)
}

df = pd.DataFrame(sample_data)

print("Data preprocessing...")

# Encode 'Response' column: 'No' -> 0, 'Yes' -> 1
df['Response'] = df['Response'].map({'No': 0, 'Yes': 1})

# Convert boolean columns to int (0 or 1)
bool_columns = df.select_dtypes(include=['bool']).columns
df[bool_columns] = df[bool_columns].astype(int)

# Create the 'Risk' column
df['Risk'] = (df['Total Claim Amount'] > df['Total Claim Amount'].median()).astype(int)

print(f"Risk distribution: {df['Risk'].value_counts()}")

# Drop irrelevant columns
df.drop(columns=["Customer", "State", "Effective To Date", "Gender",
                 "Policy Type", "Policy", "Renew Offer Type", "Sales Channel"], inplace=True)

# One-hot encode categorical variables
categorical_columns = ["Coverage", "Education", "EmploymentStatus",
                       "Marital Status", "Location Code", "Vehicle Class", "Vehicle Size"]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Convert boolean columns to int (if any remain)
bool_columns = df.select_dtypes(include=['bool']).columns
df[bool_columns] = df[bool_columns].astype(int)

# Select numerical columns for scaling
numerical_columns = ["Customer Lifetime Value", "Income", "Monthly Premium Auto",
                     "Months Since Last Claim", "Months Since Policy Inception",
                     "Number of Open Complaints", "Number of Policies", "Total Claim Amount"]

# Normalize numerical features
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print(f"Final dataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")

# Features and target
X = df.drop(columns=["Risk"])  # Features
y = df["Risk"]  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Create a pipeline that imputes missing values and then fits logistic regression
model = make_pipeline(
    SimpleImputer(strategy='mean'),
    LogisticRegression(random_state=42, max_iter=1000)
)

print("Training model...")
# Fit the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
print("Saving model and scaler...")
joblib.dump(model, "../logistic_regression_model.pkl")
joblib.dump(scaler, "../scaler.pkl")

# Save feature names and preprocessing info
feature_info = {
    'feature_names': list(X.columns),
    'numerical_columns': numerical_columns,
    'categorical_columns': categorical_columns,
    'n_features': X.shape[1]
}
joblib.dump(feature_info, "../feature_info.pkl")

print("Model training completed successfully!")
print("Files saved:")
print("- logistic_regression_model.pkl")
print("- scaler.pkl")
print("- feature_info.pkl")
