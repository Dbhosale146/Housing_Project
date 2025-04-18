from collections import namedtuple
from datetime import datetime
import uuid
from housing.config.configuration import Configuartion
from housing.logger import logging, get_log_file_name
from housing.exception import HousingException
from threading import Thread
from typing import List
from multiprocessing import Process
from housing.entity.artifact_entity import ModelPusherArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from housing.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact
from housing.entity.config_entity import DataIngestionConfig, ModelEvaluationConfig
from housing.component.data_ingestion import DataIngestion
from housing.component.data_validation import DataValidation
from housing.component.data_transformation import DataTransformation
from housing.component.model_trainer import ModelTrainer
from housing.component.model_evaluation import ModelEvaluation
from housing.component.model_pusher import ModelPusher
import os
import sys
import pandas as pd
from housing.constant import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME

# Named tuple to store experiment metadata
Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp", "artifact_time_stamp",
                                       "running_status", "start_time", "stop_time", "execution_time", "message",
                                       "experiment_file_path", "accuracy", "is_model_accepted"])


class Pipeline(Thread):
    """A threaded pipeline for executing the housing price prediction workflow."""

    # Initialize static experiment variable with default None values
    experiment: Experiment = Experiment(*([None] * 11))
    # Static variable for experiment file path
    experiment_file_path = None

    def __init__(self, config: Configuartion) -> None:
        """Initialize the Pipeline with configuration settings.

        Args:
            config (Configuartion): Configuration object containing pipeline settings.

        Raises:
            HousingException: If an error occurs during initialization.
        """
        try:
            # Create artifact directory for storing pipeline outputs
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            # Set the experiment file path
            Pipeline.experiment_file_path = os.path.join(config.training_pipeline_config.artifact_dir,
                                                        EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            # Initialize the Thread parent class
            super().__init__(daemon=False, name="pipeline")
            # Store the configuration object
            self.config = config
        except Exception as e:
            raise HousingException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """Initiate the data ingestion process.

        Returns:
            DataIngestionArtifact: Artifact containing data ingestion results.

        Raises:
            HousingException: If an error occurs during data ingestion.
        """
        try:
            # Create DataIngestion instance with configuration
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            # Execute data ingestion
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise HousingException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """Initiate the data validation process.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact from data ingestion.

        Returns:
            DataValidationArtifact: Artifact containing data validation results.

        Raises:
            HousingException: If an error occurs during data validation.
        """
        try:
            # Create DataValidation instance with configuration and artifact
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)
            # Execute data validation
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise HousingException(e, sys) from e

    def start_data_transformation(self,
                                 data_ingestion_artifact: DataIngestionArtifact,
                                 data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """Initiate the data transformation process.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact from data ingestion.
            data_validation_artifact (DataValidationArtifact): Artifact from data validation.

        Returns:
            DataTransformationArtifact: Artifact containing data transformation results.

        Raises:
            HousingException: If an error occurs during data transformation.
        """
        try:
            # Create DataTransformation instance with configuration and artifacts
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            # Execute data transformation
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise HousingException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """Initiate the model training process.

        Args:
            data_transformation_artifact (DataTransformationArtifact): Artifact from data transformation.

        Returns:
            ModelTrainerArtifact: Artifact containing model training results.

        Raises:
            HousingException: If an error occurs during model training.
        """
        try:
            # Create ModelTrainer instance with configuration and artifact
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),
                                         data_transformation_artifact=data_transformation_artifact)
            # Execute model training
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise HousingException(e, sys) from e

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                              data_validation_artifact: DataValidationArtifact,
                              model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """Initiate the model evaluation process.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact from data ingestion.
            data_validation_artifact (DataValidationArtifact): Artifact from data validation.
            model_trainer_artifact (ModelTrainerArtifact): Artifact from model training.

        Returns:
            ModelEvaluationArtifact: Artifact containing model evaluation results.

        Raises:
            HousingException: If an error occurs during model evaluation.
        """
        try:
            # Create ModelEvaluation instance with configuration and artifacts
            model_eval = ModelEvaluation(
                model_evaluation_config=self.config.get_model_evaluation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact)
            # Execute model evaluation
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise HousingException(e, sys) from e

    def start_model_pusher(self, model_eval_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """Initiate the model pushing process.

        Args:
            model_eval_artifact (ModelEvaluationArtifact): Artifact from model evaluation.

        Returns:
            ModelPusherArtifact: Artifact containing model pushing results.

        Raises:
            HousingException: If an error occurs during model pushing.
        """
        try:
            # Create ModelPusher instance with configuration and artifact
            model_pusher = ModelPusher(
                model_pusher_config=self.config.get_model_pusher_config(),
                model_evaluation_artifact=model_eval_artifact
            )
            # Execute model pushing
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise HousingException(e, sys) from e

    def run_pipeline(self):
        """Execute the full housing price prediction pipeline.

        This method orchestrates data ingestion, validation, transformation, model training,
        evaluation, and pushing, while tracking experiment metadata.

        Returns:
            Experiment: Named tuple containing the experiment's metadata.

        Raises:
            HousingException: If an error occurs during pipeline execution.
        """
        try:
            # Check if a pipeline is already running
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")
                return Pipeline.experiment
            # Log pipeline start
            logging.info("Pipeline starting.")

            # Generate unique experiment ID
            experiment_id = str(uuid.uuid4())

            # Initialize experiment metadata
            Pipeline.experiment = Experiment(experiment_id=experiment_id,
                                             initialization_timestamp=self.config.time_stamp,
                                             artifact_time_stamp=self.config.time_stamp,
                                             running_status=True,
                                             start_time=datetime.now(),
                                             stop_time=None,
                                             execution_time=None,
                                             experiment_file_path=Pipeline.experiment_file_path,
                                             is_model_accepted=None,
                                             message="Pipeline has been started.",
                                             accuracy=None)
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")

            # Save initial experiment state
            self.save_experiment()

            # Execute pipeline steps
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_validation_artifact=data_validation_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)

            # Push model if accepted
            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_eval_artifact=model_evaluation_artifact)
                logging.info(f'Model pusher artifact: {model_pusher_artifact}')
            else:
                logging.info("Trained model rejected.")
            logging.info("Pipeline completed.")

            # Record pipeline completion time
            stop_time = datetime.now()
            # Update experiment metadata
            Pipeline.experiment = Experiment(experiment_id=Pipeline.experiment.experiment_id,
                                             initialization_timestamp=self.config.time_stamp,
                                             artifact_time_stamp=self.config.time_stamp,
                                             running_status=False,
                                             start_time=Pipeline.experiment.start_time,
                                             stop_time=stop_time,
                                             execution_time=stop_time - Pipeline.experiment.start_time,
                                             message="Pipeline has been completed.",
                                             experiment_file_path=Pipeline.experiment_file_path,
                                             is_model_accepted=model_evaluation_artifact.is_model_accepted,
                                             accuracy=model_trainer_artifact.model_accuracy)
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            # Save final experiment state
            self.save_experiment()
        except Exception as e:
            raise HousingException(e, sys) from e

    def run(self):
        """Run the pipeline as a thread.

        This method is called when the thread is started.

        Raises:
            Exception: If an error occurs during pipeline execution.
        """
        try:
            self.run_pipeline()
        except Exception as e:
            raise e

    def save_experiment(self):
        """Save the current experiment metadata to a CSV file.

        The experiment data is stored in a CSV file, appending new entries if the file exists.

        Raises:
            HousingException: If an error occurs while saving the experiment.
        """
        try:
            if Pipeline.experiment.experiment_id is not None:
                # Convert experiment named tuple to dictionary
                experiment = Pipeline.experiment
                experiment_dict = experiment._asdict()
                # Convert values to lists for DataFrame compatibility
                experiment_dict: dict = {key: [value] for key, value in experiment_dict.items()}

                # Add additional metadata
                experiment_dict.update({
                    "created_time_stamp": [datetime.now()],
                    "experiment_file_path": [os.path.basename(Pipeline.experiment.experiment_file_path)]})

                # Create DataFrame from experiment data
                experiment_report = pd.DataFrame(experiment_dict)

                # Create directory for experiment file
                os.makedirs(os.path.dirname(Pipeline.experiment_file_path), exist_ok=True)
                # Append to existing file or create new one
                if os.path.exists(Pipeline.experiment_file_path):
                    experiment_report.to_csv(Pipeline.experiment_file_path, index=False, header=False, mode="a")
                else:
                    experiment_report.to_csv(Pipeline.experiment_file_path, mode="w", index=False, header=True)
            else:
                print("First start experiment")
        except Exception as e:
            raise HousingException(e, sys) from e

    @classmethod
    def get_experiments_status(cls, limit: int = 5) -> pd.DataFrame:
        """Retrieve the status of recent experiments.

        Args:
            limit (int, optional): Number of recent experiments to return. Defaults to 5.

        Returns:
            pd.DataFrame: DataFrame containing experiment metadata, excluding certain columns.

        Raises:
            HousingException: If an error occurs while reading the experiment file.
        """
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                # Read experiment data from CSV
                df = pd.read_csv(Pipeline.experiment_file_path)
                # Return the most recent experiments, excluding specified columns
                limit = -1 * int(limit)
                return df[limit:].drop(columns=["experiment_file_path", "initialization_timestamp"], axis=1)
            else:
                # Return empty DataFrame if file doesn't exist
                return pd.DataFrame()
        except Exception as e:
            raise HousingException(e, sys) from e