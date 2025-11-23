from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import sys

app = Flask(__name__)

# Initialize variables
model = None
scaler = None
le_dict = None

# Load models with better error handling
def load_models():
    global model, scaler, le_dict
    try:
        print("Loading logistic_model.pkl...", file=sys.stderr)
        model = pickle.load(open('logistic_model.pkl', 'rb'))
        print("Loading scaler.pkl...", file=sys.stderr)
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        print("Loading labelencoders.pkl...", file=sys.stderr)
        with open("labelencoders.pkl", "rb") as f:
            le_dict = pickle.load(f)
        print("All models loaded successfully!", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}", file=sys.stderr)
        sys.stderr.flush()
        raise e
    except Exception as e:
        print(f"ERROR LOADING MODEL: {e}", file=sys.stderr)
        sys.stderr.flush()
        raise e

# Load models at startup
load_models()

# Urutan fitur HARUS sama dengan training
feature_order = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "FamilyHistory",
    "HighCaloricFood",
    "Vegetables",
    "MainMeals",
    "Snacks",
    "Smoking",
    "Alcohol",
    "Water",
    "Monitor",
    "Exercise",
    "Devices",
    "Transport"
]

@app.route("/health")
def health():
    """Health check endpoint"""
    if model is None or scaler is None or le_dict is None:
        return jsonify({"status": "error", "message": "Models not loaded"}), 500
    return jsonify({"status": "ok", "message": "Application is running"}), 200

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            input_data = []

            for feature in feature_order:
                value = request.form[feature]

                # Jika kolom kategori → encode pakai LabelEncoder
                if feature in le_dict:
                    value = le_dict[feature].transform([value])[0]
                else:
                    value = float(value)

                input_data.append(value)

            # Convert to array
            input_data = np.array([input_data])

            # Scaling
            input_scaled = scaler.transform(input_data)

            # Predict
            pred_num = model.predict(input_scaled)[0]

            # Convert label number → original label
            result_label = le_dict["Obesity"].inverse_transform([pred_num])[0]

            result = f"Obesity Level Prediction Results: {result_label}"
        except Exception as e:
            print(f"ERROR in prediction: {e}", file=sys.stderr)
            sys.stderr.flush()
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
