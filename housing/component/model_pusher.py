from housing.logger import logging
from housing.exception import HousingException
from housing.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact 
from housing.entity.config_entity import ModelPusherConfig
import os, sys
import shutil



class ModelPusher:
    """
    The ModelPusher class is responsible for exporting the evaluated model
    to a defined export directory for deployment or production use.
    """

    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact
                 ):
        """
        Initialize ModelPusher with configuration and evaluated model artifact.

        Parameters:
        - model_pusher_config: Contains the path for exporting the model.
        - model_evaluation_artifact: Contains the path of the evaluated model to be exported.
        """
        try:
            logging.info(f"{'>>' * 30}Model Pusher log started.{'<<' * 30} ")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            raise HousingException(e, sys) from e

    def export_model(self) -> ModelPusherArtifact:
        """
        Copies the evaluated model file to the export directory.

        Returns:
        - ModelPusherArtifact: Contains status and the path where the model is exported.
        """
        try:
            evaluated_model_file_path = self.model_evaluation_artifact.evaluated_model_path
            export_dir = self.model_pusher_config.export_dir_path
            model_file_name = os.path.basename(evaluated_model_file_path)
            export_model_file_path = os.path.join(export_dir, model_file_name)
            logging.info(f"Exporting model file: [{export_model_file_path}]")
            os.makedirs(export_dir, exist_ok=True)

            shutil.copy(src=evaluated_model_file_path, dst=export_model_file_path)

            # Optional: Save model to cloud services like Azure, GCP, or AWS S3

            logging.info(
                f"Trained model: {evaluated_model_file_path} is copied in export dir:[{export_model_file_path}]")

            model_pusher_artifact = ModelPusherArtifact(is_model_pusher=True,
                                                        export_model_file_path=export_model_file_path
                                                        )
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            return model_pusher_artifact
        except Exception as e:
            raise HousingException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Initiates the model pushing process.

        Returns:
        - ModelPusherArtifact: The result of the export operation.
        """
        try:
            return self.export_model()
        except Exception as e:
            raise HousingException(e, sys) from e

    def __del__(self):
        """
        Destructor to log the end of model pusher execution.
        """
        logging.info(f"{'>>' * 20}Model Pusher log completed.{'<<' * 20} ")
