import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass   #Automatically adds a constructor to this class — no need to write __init__
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("enterd the data ingestion method or component")
        try:
            df=pd.read_csv('Notebook\\data\\cleanedFoodDelivery.csv')
            logging.info("read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)   #this line checks wheather the artifacts folderexist or not if not then it create that folder and inside this it create train.csv file 

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train test split initiated")
            # doing the train test split
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion of the data has been completed")

            # return the path of the test and train files where data is stored
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)



if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()   #get the file path of train and test data

    data_transformation=DataTransformation()
    
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    # do the preprocessing and save the preprocsseor for later use
    # it return 3 things but we need only first two

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))



