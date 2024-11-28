from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and preprocessor (if any)
MODEL_PATH = "../models/random_forest_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    """
    Home endpoint to verify the API is running.
    """
    return jsonify({"message": "Hospital Readmission Prediction API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to make predictions based on input data.
    
    Expected Input (JSON):
    {
        "features": [
            {
                "age": 50,
                "bmi": 24.5,
                "gender_male": 1,
                "gender_female": 0,
                ...
            }
        ]
    }
    
    Returns:
        JSON: Predictions for each input instance.
    """
    try:
        # Parse input data
        input_data = request.get_json()
        if "features" not in input_data:
            return jsonify({"error": "Missing 'features' key in JSON input."}), 400
        
        # Convert input data to DataFrame
        features = pd.DataFrame(input_data["features"])
        
        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]
        
        # Prepare response
        results = [
            {"prediction": int(pred), "probability": round(float(prob), 4)}
            for pred, prob in zip(predictions, probabilities)
        ]
        
        return jsonify({"predictions": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
