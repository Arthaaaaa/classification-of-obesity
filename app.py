from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model dan scaler
model = pickle.load(open('logistic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Mapping kategori → angka (HARUS SAMA dengan training)
map_gender = {"Male": 1, "Female": 0}
map_bin = {"yes": 1, "no": 0}

map_high_calorie = {"yes": 1, "no": 0}

map_vegetables = {
    "never": 0,
    "sometimes": 1,
    "always": 2
}

map_main_meals = {
    "1-2": 0,
    "3": 1,
    "more than 3": 2
}

map_snacks = {
    "no": 0,
    "sometimes": 1,
    "frequently": 2,
    "always": 3
}

map_smoke = {"yes": 1, "no": 0}

map_alcohol = {
    "no": 0,
    "sometimes": 1,
    "frequently": 2,
    "always": 3
}

map_water = {
    "1-2L": 0,
    "more than 2L": 1
}

map_monitor = {"yes": 1, "no": 0}

map_exercise = {
    "none": 0,
    "1-2 days": 1,
    "2-4 days": 2,
    "4-5 days": 3,
    "almost every day": 4
}

map_devices = {
    "0-2 hours": 0,
    "3-5 hours": 1,
    "more than 5 hours": 2
}

map_transport = {
    "car": 0,
    "motorcycle": 1,
    "public": 2,
    "walking": 3
}

# Mapping hasil prediksi
label_map = {
    0: "insufficient_weight",
    1: "normal_weight",
    5: "overweight",
    2: "obesity_type_I",
    3: "obesity_type_II",
    4: "obesity_type_III"
}


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == "POST":

        # Ambil nilai dari form
        data = [
            map_gender[request.form['gender']],
            float(request.form['age']),
            float(request.form['height']),
            float(request.form['weight']),
            map_bin[request.form['family']],
            map_high_calorie[request.form['high_calorie'].lower()],
            map_vegetables[request.form['vegetables']],
            map_main_meals[request.form['main_meals']],
            map_snacks[request.form['snacks']],
            map_smoke[request.form['smoke']],
            map_alcohol[request.form['alcohol']],
            map_water[request.form['water']],
            map_monitor[request.form['monitor']],
            map_exercise[request.form['exercise']],
            map_devices[request.form['devices']],
            map_transport[request.form['transport']]
        ]

        data = np.array([data])

        # Scaling
        data_scaled = scaler.transform(data)

        # Prediksi
        pred_num = model.predict(data_scaled)[0]
        pred_label = label_map[pred_num]

        result = f"Obesity Level Prediction Results: {pred_label}"

    return render_template('index.html', result=result)



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
