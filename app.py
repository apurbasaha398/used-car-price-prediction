from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    data = CustomData(
        back_legroom=float(request.form.get('back_legroom')) if request.form.get('back_legroom') else np.nan,
        body_type=request.form.get('body_type'),
        num_of_cylinders=int(request.form.get('num_of_cylinders')) if request.form.get('num_of_cylinders') else np.nan,
        exterior_color_grouped=request.form.get('exterior_color_grouped'),
        front_legroom=float(request.form.get('front_legroom')) if request.form.get('front_legroom') else np.nan,
        fuel_tank_volume=float(request.form.get('fuel_tank_volume')) if request.form.get('fuel_tank_volume') else np.nan,
        fuel_type=request.form.get('fuel_type'),
        horsepower=float(request.form.get('horsepower')) if request.form.get('horsepower') else np.nan,
        interior_color_grouped=request.form.get('interior_color_grouped'),
        make_name=request.form.get('make_name'),
        maximum_seating=int(request.form.get('maximum_seating')) if request.form.get('maximum_seating') else np.nan,
        mileage=request.form.get('mileage'),
        model_name=request.form.get('model_name'),
        seller_rating=request.form.get('seller_rating'),
        torque=float(request.form.get('torque')) if request.form.get('torque') else np.nan,
        transmission=request.form.get('transmission'),
        wheel_system=request.form.get('wheel_system'),
        wheelbase=float(request.form.get('wheelbase')) if request.form.get('wheelbase') else np.nan,
        volume=float(request.form.get('volume')) if request.form.get('volume') else np.nan,
        year=int(request.form.get('year')),
        combined_fuel_economy=float(request.form.get('combined_fuel_economy')) if request.form.get('combined_fuel_economy') else np.nan,
        dealer_region=request.form.get('dealer_region') if request.form.get('dealer_region') else np.nan
    )
    pred_df = data.get_data_as_frame()
    #print(pred_df)
    
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return render_template('home.html', results=results)

if __name__ == '__main__':
    app.static_folder = 'static'
    app.run(host='0.0.0.0', port=8080)
