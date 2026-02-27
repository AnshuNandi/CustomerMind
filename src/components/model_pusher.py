import sys
import shutil
import os

from src.entity.artifact_entity import (ModelPusherArtifact,
                                           ModelTrainerArtifact)
from src.entity.config_entity import ModelPusherConfig
from src.exception import CustomerException
from src.logger import logging


class ModelPusher:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("Saving model artifacts to local storage")
            # Create model directory if it doesn't exist
            os.makedirs(self.model_pusher_config.model_export_dir, exist_ok=True)
            
            # Copy trained model to export directory
            model_export_path = os.path.join(
                self.model_pusher_config.model_export_dir,
                os.path.basename(self.model_trainer_artifact.trained_model_file_path)
            )
            shutil.copy(
                self.model_trainer_artifact.trained_model_file_path,
                model_export_path
            )
            logging.info(f"Model saved to: {model_export_path}")
            
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=model_export_path,
            )
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e
