from pandas import DataFrame
from sklearn.pipeline import Pipeline
from src.exception import CustomerException
from src.logger import logging
import os, sys
import pickle

from dataclasses import dataclass




class CustomerSegmentationModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        logging.info("Entered predict method of srcTruckModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise CustomerException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class CustomerClusterEstimator:
    """
    This class is used to load the saved model and check if model exists
    """
    def __init__(self, bucket_name: str, model_path: str):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.loaded_model = None

    def is_model_present(self, model_path: str) -> bool:
        """
        Check if model file exists in the specified path
        """
        try:
            return os.path.exists(model_path)
        except Exception as e:
            raise CustomerException(e, sys) from e

    def load_model(self) -> CustomerSegmentationModel:
        """
        Load the saved model from file
        """
        try:
            if self.is_model_present(self.model_path):
                with open(self.model_path, 'rb') as file_obj:
                    self.loaded_model = pickle.load(file_obj)
                logging.info(f"Model loaded successfully from: {self.model_path}")
                return self.loaded_model
            else:
                raise Exception(f"Model file not found at path: {self.model_path}")
        except Exception as e:
            raise CustomerException(e, sys) from e

    def predict(self, dataframe: DataFrame):
        """
        Make predictions using the loaded model
        """
        try:
            if self.loaded_model is None:
                self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            raise CustomerException(e, sys) from e
