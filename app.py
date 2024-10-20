import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

ridge_model = pickle.load(open("models/rigid.pkl", "rb"))
scaler_model = pickle.load(open("models/scaler.pkl", "rb"))

@app.route('/predict_datapoint', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == "POST":
        # Retrieve form data
        Temperture = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Create input array
        input_data = np.array([[Temperture, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        # Scale the input data using the scaler_model
        new_data_scaled = scaler_model.transform(input_data)

        # Make prediction using the ridge model
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result=result[0])
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)
 