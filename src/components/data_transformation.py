# here in this we can do feature engineering, data cleaning ,data tranformation

import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        # this function is responsible for data transformation...

        try:
            numerical_columns=['Distance_km','Preparation_Time_min']
            categorical_columns=['Weather', 'Traffic_Level','Time_of_Day']

            preprocessor=ColumnTransformer([
                ("ohe",OneHotEncoder(),categorical_columns),
                ("scaling",StandardScaler(),numerical_columns)
            ])

            logging.info("created the preprocessor")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            tarin_df=train_df.drop(columns=['Unnamed: 0'],axis=1)

            test_df=pd.read_csv(test_path)
            test_df=test_df.drop(columns=['Unnamed: 0'],axis=1)

            logging.info("reading the train and test data completed")

            logging.info("obtaining preprocessor object")

            preprocessor_obj=self.get_data_transformer_object()

            target_column_name='Delivery_Time_min'
            numerical_columns=['Distance_km','Preparation_Time_min']
            categorical_columns=['Weather', 'Traffic_Level','Time_of_Day']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("applying preprocssing object on the traing and testing data")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            # np.c_[] is used to concatenate arrays column-wise (i.e., along axis=1
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            # Saves the preprocessor object to use later (e.g., in prediction).
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

