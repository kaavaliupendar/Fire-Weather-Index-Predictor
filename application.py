import os
import pickle
from flask import Flask, request, render_template
import numpy as np

application = Flask(__name__)
app = application

# Load pre-trained model and scaler
try:
    model_path = 'models/ridge.pkl'
    scaler_path = 'models/scaler.pkl'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    ridge_model = pickle.load(open(model_path, 'rb'))
    standard_scaler = pickle.load(open(scaler_path, 'rb'))

except FileNotFoundError as e:
    raise RuntimeError(f"Model or scaler file not found: {e}")

# Route: Home/Landing page
@app.route('/')
def index():
    return render_template('index.html')

# Route: Prediction page
@app.route('/home', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Input fields expected from form
            input_fields = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
            data = []

            # Parse and validate inputs
            for field in input_fields:
                value = request.form.get(field)
                if value is None or value.strip() == "":
                    raise ValueError(f"Missing value for {field}")
                try:
                    data.append(float(value))
                except ValueError:
                    raise ValueError(f"Invalid numeric value for {field}: {value}")

            # Prepare data and predict
            scaled_input = standard_scaler.transform([data])
            prediction = ridge_model.predict(scaled_input)[0]

            return render_template('home.html', results=round(float(prediction), 2))

        except Exception as e:
            return render_template('home.html', results=f"Error: {e}")

    return render_template('home.html', results=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
