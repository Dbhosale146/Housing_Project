from housing.logger import logging
from housing.exception import HousingException
from housing.entity.config_entity import DataValidationConfig
from housing.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
import os,sys
import pandas  as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import json


class DataValidation:
    """
    This class handles the validation of the dataset:
    - Checks for the existence of training and testing files.
    - Validates schema (column names, value types).
    - Performs data drift analysis using the 'evidently' library.
    """

    def __init__(self, data_validation_config:DataValidationConfig,
        data_ingestion_artifact:DataIngestionArtifact):
        """
        Initializes the DataValidation object with config and artifact paths.

        Args:
        data_validation_config (DataValidationConfig): Config for paths and report files.
        data_ingestion_artifact (DataIngestionArtifact): Paths for train and test CSVs.
        """
        try:
            logging.info(f"{'>>'*30}Data Valdaition log started.{'<<'*30} \n\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    def get_train_and_test_df(self):
        """
        Reads the training and testing dataframes from the ingested CSV paths.

        Returns:
        Tuple of (train_df, test_df)
        """
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df,test_df
        except Exception as e:
            raise HousingException(e,sys) from e

    def is_train_test_file_exists(self)->bool:
        """
        Verifies whether the train and test CSV files exist in the expected location.

        Returns:
        bool: True if both files exist, else raises an exception.
        """
        try:
            logging.info ("Checking if training and test file is available")
            is_train_file_exist = False
            is_test_file_exist = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path)

            is_available =  is_train_file_exist and is_test_file_exist

            logging.info(f"Is train and test file exists?-> {is_available}")
            
            if not is_available:
                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path
                message=f"Training file: {training_file} or Testing file: {testing_file}" \
                    "is not present"
                raise Exception(message)

            return is_available
        except Exception as e:
            raise HousingException(e,sys) from e

    def validate_dataset_schema(self)->bool:
        """
        Placeholder for validating schema of the dataset.
        (To be implemented: check column names, data types, and allowed values.)

        Returns:
        bool: True (currently hardcoded)
        """
        try:
            validation_status = False
            # Assignment: validate training and testing dataset using schema file
            validation_status = True
            return validation_status 
        except Exception as e:
            raise HousingException(e,sys) from e

    def get_and_save_data_drift_report(self):
        """
        Uses Evidently to generate a data drift report in JSON format between train and test sets.

        Returns:
        Report object (evidently.report.Report)
        """
        try:
            report = Report(metrics=[DataDriftPreset()])
            train_df, test_df = self.get_train_and_test_df()
            report.run(reference_data=train_df, current_data=test_df)

            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            report.save_json(report_file_path)

            return report
        except Exception as e:
            raise HousingException(e, sys) from e

    def save_data_drift_report_page(self):
        """
        Saves the HTML version of the data drift report to the path defined in the config.
        """
        try:
            report = Report(metrics=[DataDriftPreset()])
            train_df, test_df = self.get_train_and_test_df()
            report.run(reference_data=train_df, current_data=test_df)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir, exist_ok=True)

            report.save_html(report_page_file_path)
        except Exception as e:
            raise HousingException(e, sys) from e

    def is_data_drift_found(self)->bool:
        """
        Triggers data drift detection and saves reports.

        Returns:
        bool: True if successful
        """
        try:
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise HousingException(e,sys) from e

    def initiate_data_validation(self)->DataValidationArtifact:
        """
        Coordinates the entire data validation process and returns a DataValidationArtifact.

        Returns:
        DataValidationArtifact
        """
        try:
            self.is_train_test_file_exists()
            self.validate_dataset_schema()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=True,
                message="Data Validation performed successully."
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    def __del__(self):
        """
        Destructor that logs when data validation is completed.
        """
        logging.info(f"{'>>'*30}Data Valdaition log completed.{'<<'*30} \n\n")
