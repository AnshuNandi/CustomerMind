from src.ml.model.estimator import CustomerSegmentationModel, CustomerClusterEstimator
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig , ModelTrainerConfig
from src.constant.training_pipeline import *
from src.entity.config_entity import training_pipeline_config
from src.entity.config_entity import Prediction_config, PredictionPipelineConfig

from src.entity.config_entity import DataTransformationConfig , ModelTrainerConfig
from src.logger import logging
from src.utils.main_utils import MainUtils

from src.exception import CustomerException
import pandas as pd
import numpy as np
import sys
import os
import glob

import logging
import sys
from pandas import DataFrame
import pandas as pd





class CustomerData:
    def __init__(self):
        pass
        
    def get_input_dataset(self, column_schema:dict, input_data):
        columns = column_schema.keys()
        
        input_dataset = pd.DataFrame([input_data], columns = columns)
        for key, value in column_schema.items():
            input_dataset[key] = input_dataset[key].astype(value)
        
        return input_dataset

    @staticmethod
    def form_input_dataframe(data):
        prediction_config = Prediction_config()
        prediction_schema = prediction_config.__dict__
        column_schema = prediction_schema['prediction_schema']['columns']

        customerData = CustomerData()
        input_dataset = customerData.get_input_dataset(
            column_schema=column_schema,
            input_data=data
        )
        
        return input_dataset
        
        
    


class PredictionPipeline:
    def __init__(self):
        self.utils = MainUtils()
        
    def prepare_input_data(self, input_data:list) -> pd.DataFrame:
        """ 
        method: prepare_input_data 
        
        objective: This method creates a dataframe taking the column names from prediction schema file
                       with the input values for prediction and returns it

        Args:
            input_data (list): input data 

        Raises:
            CustomerException

        Returns:
            customerDataframe: pd.DataFrame: a dataframe containing the input values
        """
        try:
        
            
            customerDataframe = CustomerData.form_input_dataframe(data = input_data)
            logging.info("customerDatafram has been created")
            return customerDataframe
        except Exception as e:
            raise CustomerException(e,sys)
        
   
        
    
        
    def get_trained_model(self, ModelTrainerConfig = ModelTrainerConfig):
        """
        method: get_trained_model
        
        objective: this method returns the model

        Args:
            ModelTrainerConfig

        Raises:
            CustomerException: 

        Returns:
            model: latest trained model
        """
        try:
            # Look for the most recent trained model in the artifact directory
            import glob
            artifact_dir = "src/artifact"
            
            # Find all model.pkl files in artifact directory
            model_files = glob.glob(f"{artifact_dir}/**/model.pkl", recursive=True)
            
            if model_files:
                # Get the most recently trained model
                latest_model_path = max(model_files, key=os.path.getctime)
                logging.info(f"Loading model from: {latest_model_path}")
                
                model = CustomerClusterEstimator(
                    bucket_name="customer-segmentation-models",
                    model_path=latest_model_path
                )
                return model
            else:
                # Fallback to default if no model found
                logging.warning("No trained model found in artifacts. Train the model first using /train endpoint")
                raise Exception("No trained model found. Please train the model first by visiting /train endpoint")
                
        except Exception as e:
            raise CustomerException(e, sys) from e
        
    def run_pipeline(self, input_data:list):
        
        """
        method: run_pipeline
        
        objective: run_pipeline method runs the whole prediction pipeline.

        Raises:
            CustomerException: 
        """
        try:
            input_dataframe =  self.prepare_input_data(input_data) 
            model = self.get_trained_model()
            prediction = model.predict(input_dataframe)
            return prediction
            
        except Exception as e:
            raise CustomerException(e, sys)
            
            
        
            
        

 
        

        