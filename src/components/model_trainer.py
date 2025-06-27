# here we write the code for model training....
import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=0.1),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Support Vector Regressor': SVR()
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # toget the best models score from the dict
            best_model_score=max(sorted(model_report.values()))

            # to get the best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.7:
                raise CustomException("No best model found")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_score_best_model=r2_score(y_test,predicted)

            return r2_score_best_model

        except Exception as e:
            raise CustomException(e,sys)