from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

# route for the home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    distance = float(request.form.get('distance_km'))
    prep_time = float(request.form.get('food_preptime'))

        # Constraint checks
    if distance > 25:
        return render_template("index.html", error="Distance must be less than 25 km.")
    if prep_time > 30:
        return render_template("index.html", error="Preparation time must be less than 30 minutes.")


    data=CustomData(distance=request.form.get('distance_km'),
                    weather_type=request.form.get('weather'),
                    traffic_level=request.form.get('traffic'),
                    timeofday=request.form.get('timeofday'),
                    food_prep_time=request.form.get('food_preptime'))
    pred_df=data.get_data_as_dataframe()
    print(pred_df)

    predict_pipline=PredictPipeline()
    results=predict_pipline.predict(pred_df)
    return render_template("ouput.html",result=f'The Predicted Delivery Time is {round(results[0],2)} minutes')


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
