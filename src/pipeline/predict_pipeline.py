import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,distance:float,weather_type : str,traffic_level:str,timeofday:str,food_prep_time: float):

        self.Distance_km=distance
        self.Weather=weather_type
        self.Traffic_Level=traffic_level
        self.Time_of_Day=timeofday
        self.Preparation_Time_min=food_prep_time


    def get_data_as_dataframe(self):
        try:
            custom_data_imput_dict={
                "Distance_km":[self.Distance_km],
                "Weather":[self.Weather],
                "Traffic_Level":[self.Traffic_Level],
                "Time_of_Day":[self.Time_of_Day],
                "Preparation_Time_min":[self.Preparation_Time_min]
            }

            return pd.DataFrame(custom_data_imput_dict)
        except Exception as e:
            raise CustomData(e,sys)


        