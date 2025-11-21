from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

try:
    model = pickle.load(open('logistic_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    print("ERROR LOADING MODEL:", e)
    raise e


# Load label encoders
with open("labelencoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

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

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
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

    return render_template("index.html", result=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
