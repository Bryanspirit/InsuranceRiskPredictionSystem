from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Global variables for model components
model = None
scaler = None
feature_info = None


def load_model_components():
    """Load model, scaler, and feature information"""
    global model, scaler, feature_info

    try:
        model = joblib.load("logistic_regression_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_info = joblib.load("feature_info.pkl")
        print("Model components loaded successfully!")
        print(f"Model features: {len(feature_info['feature_names'])}")
        return True
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please run the training script first to generate model files.")
        return False


def preprocess_input_data(input_data):
    """Preprocess input data to match training format"""

    # Create a DataFrame with the input data
    df = pd.DataFrame([input_data])

    # Ensure all categorical columns are present (set to 0 if not provided)
    all_features = feature_info['feature_names']

    # Initialize all features to 0
    processed_data = pd.DataFrame(0, index=[0], columns=all_features)

    # Set the numerical features
    numerical_columns = feature_info['numerical_columns']
    for col in numerical_columns:
        if col in df.columns:
            processed_data[col] = df[col].values[0]

    # Set Response if provided
    if 'Response' in df.columns:
        processed_data['Response'] = df['Response'].values[0]

    # Handle categorical features (one-hot encoded)
    # For simplicity, we'll set provided categorical features
    categorical_mappings = {
        'Coverage': ['Coverage_Extended', 'Coverage_Premium'],
        'Education': ['Education_College', 'Education_Doctor', 'Education_High School', 'Education_Master'],
        'EmploymentStatus': ['EmploymentStatus_Retired', 'EmploymentStatus_Unemployed'],
        'Marital Status': ['Marital Status_Married', 'Marital Status_Single'],
        'Location Code': ['Location Code_Suburban', 'Location Code_Urban'],
        'Vehicle Class': ['Vehicle Class_Four-Door Car', 'Vehicle Class_Luxury Car', 'Vehicle Class_SUV'],
        'Vehicle Size': ['Vehicle Size_Medsize', 'Vehicle Size_Small']
    }

    # Set categorical features based on input
    for base_col, encoded_cols in categorical_mappings.items():
        if base_col in input_data:
            value = input_data[base_col]
            # Find the corresponding encoded column
            encoded_col = f"{base_col}_{value}"
            if encoded_col in processed_data.columns:
                processed_data[encoded_col] = 1

    return processed_data


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or feature_info is None:
        return jsonify({"error": "Model not loaded. Please run training script first."}), 500

    try:
        # Get form data
        data = request.get_json()

        # Preprocess the input data
        processed_data = preprocess_input_data(data)

        # Scale numerical features
        numerical_columns = feature_info['numerical_columns']
        processed_data[numerical_columns] = scaler.transform(processed_data[numerical_columns])

        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]

        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        confidence = max(probability) * 100

        return jsonify({
            "prediction": int(prediction),
            "risk_level": risk_level,
            "confidence": round(confidence, 2),
            "probability_low": round(probability[0] * 100, 2),
            "probability_high": round(probability[1] * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 400


@app.route("/model_info", methods=["GET"])
def model_info():
    """Get information about the loaded model"""
    if feature_info is None:
        return jsonify({"error": "Model not loaded"}), 500

    return jsonify({
        "n_features": feature_info['n_features'],
        "numerical_columns": feature_info['numerical_columns'],
        "categorical_columns": feature_info['categorical_columns'],
        "feature_names": feature_info['feature_names'][:10]  # Show first 10 features
    })


if __name__ == "__main__":
    # Load model components on startup
    if load_model_components():
        print("Starting Flask application...")
        app.run(debug=True, port=5000)
    else:
        print("Failed to load model components. Please run the training script first.")
