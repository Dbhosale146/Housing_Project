from cmath import log
import importlib
from pyexpat import model
import numpy as np
import yaml
from housing.exception import HousingException
import os
import sys

from collections import namedtuple
from typing import List
from housing.logger import logging
from sklearn.metrics import r2_score, mean_squared_error

# Constants for configuration keys used in model configuration
GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"

# Named tuple to store initialized model details
InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

# Named tuple to store grid search results for a model
GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score"])

# Named tuple to store details of the best model
BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score"])

# Named tuple to store evaluation metrics for a model
MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])


def evaluate_classification_model(model_list: list, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, base_accuracy: float = 0.6) -> MetricInfoArtifact:
    """Evaluate a list of classification models and return the best model's metrics.

    Args:
        model_list (list): List of classification model objects.
        X_train (np.ndarray): Training dataset input features.
        y_train (np.ndarray): Training dataset target values.
        X_test (np.ndarray): Testing dataset input features.
        y_test (np.ndarray): Testing dataset target values.
        base_accuracy (float, optional): Minimum acceptable accuracy threshold. Defaults to 0.6.

    Returns:
        MetricInfoArtifact: Named tuple containing evaluation metrics of the best model.
    """
    pass


def evaluate_regression_model(model_list: list, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, base_accuracy: float = 0.6) -> MetricInfoArtifact:
    """Compare multiple regression models and return the best model's metrics.

    Args:
        model_list (list): List of regression model objects.
        X_train (np.ndarray): Training dataset input features.
        y_train (np.ndarray): Training dataset target values.
        X_test (np.ndarray): Testing dataset input features.
        y_test (np.ndarray): Testing dataset target values.
        base_accuracy (float, optional): Minimum acceptable accuracy threshold. Defaults to 0.6.

    Returns:
        MetricInfoArtifact: Named tuple containing evaluation metrics of the best model.

    Raises:
        HousingException: If an error occurs during model evaluation.
    """
    try:
        # Initialize index for tracking models
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            # Extract model name from model object
            model_name = str(model)
            logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")
            
            # Predict on training and testing datasets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R² scores for training and testing
            train_acc = r2_score(y_train, y_train_pred)
            test_acc = r2_score(y_test, y_test_pred)
            
            # Calculate RMSE for training and testing
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Compute harmonic mean of train and test accuracy
            model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
            diff_test_train_acc = abs(test_acc - train_acc)
            
            # Log evaluation metrics
            logging.info(f"{'>>'*30} Score {'<<'*30}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

            logging.info(f"{'>>'*30} Loss {'<<'*30}")
            logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].") 
            logging.info(f"Train root mean squared error: [{train_rmse}].")
            logging.info(f"Test root mean squared error: [{test_rmse}].")

            # Check if model meets accuracy and consistency criteria
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                         model_object=model,
                                                         train_rmse=train_rmse,
                                                         test_rmse=test_rmse,
                                                         train_accuracy=train_acc,
                                                         test_accuracy=test_acc,
                                                         model_accuracy=model_accuracy,
                                                         index_number=index_number)

                logging.info(f"Acceptable model found {metric_info_artifact}. ")
            index_number += 1
        if metric_info_artifact is None:
            logging.info(f"No model found with higher accuracy than base accuracy")
        return metric_info_artifact
    except Exception as e:
        raise HousingException(e, sys) from e


def get_sample_model_config_yaml_file(export_dir: str):
    """Generate a sample model configuration YAML file.

    Args:
        export_dir (str): Directory path to save the YAML file.

    Returns:
        str: Path to the generated YAML file.

    Raises:
        HousingException: If an error occurs during file creation.
    """
    try:
        # Define sample model configuration
        model_config = {
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY: "GridSearchCV",
                PARAM_KEY: {
                    "cv": 3,
                    "verbose": 1
                }
            },
            MODEL_SELECTION_KEY: {
                "module_0": {
                    MODULE_KEY: "module_of_model",
                    CLASS_KEY: "ModelClassName",
                    PARAM_KEY: {
                        "param_name1": "value1",
                        "param_name2": "value2",
                    },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ['param_value_1', 'param_value_2']
                    }
                },
            }
        }
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        # Write configuration to YAML file
        with open(export_file_path, 'w') as file:
            yaml.dump(model_config, file)
        return export_file_path
    except Exception as e:
        raise HousingException(e, sys)


class ModelFactory:
    """Factory class for initializing and managing machine learning models."""

    def __init__(self, model_config_path: str = None):
        """Initialize ModelFactory with a configuration file.

        Args:
            model_config_path (str, optional): Path to the model configuration YAML file.

        Raises:
            HousingException: If an error occurs during initialization.
        """
        try:
            # Load configuration from YAML file
            self.config: dict = ModelFactory.read_params(model_config_path)

            # Extract grid search configuration
            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])

            # Extract model selection configuration
            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])

            # Initialize lists for storing model details
            self.initialized_model_list = None
            self.grid_searched_best_model_list = None

        except Exception as e:
            raise HousingException(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref: object, property_data: dict):
        """Update properties of a class instance with provided data.

        Args:
            instance_ref (object): Instance of the class to update.
            property_data (dict): Dictionary containing property names and values.

        Returns:
            object: Updated class instance.

        Raises:
            HousingException: If property_data is not a dictionary or an error occurs.
        """
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")
            print(property_data)
            for key, value in property_data.items():
                logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise HousingException(e, sys) from e

    @staticmethod
    def read_params(config_path: str) -> dict:
        """Read configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: Configuration data.

        Raises:
            HousingException: If an error occurs during file reading.
        """
        try:
            with open(config_path) as yaml_file:
                config: dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise HousingException(e, sys) from e

    @staticmethod
    def class_for_name(module_name: str, class_name: str):
        """Dynamically load a class from a module.

        Args:
            module_name (str): Name of the module containing the class.
            class_name (str): Name of the class to load.

        Returns:
            type: Reference to the class.

        Raises:
            HousingException: If the module or class cannot be loaded.
        """
        try:
            # Load the module
            module = importlib.import_module(module_name)
            # Get the class reference
            logging.info(f"Executing command: from {module} import {class_name}")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise HousingException(e, sys) from e

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, input_feature,
                                     output_feature) -> GridSearchedBestModel:
        """Perform grid search to find the best model parameters.

        Args:
            initialized_model (InitializedModelDetail): Model details including model and parameter grid.
            input_feature: Input features for training.
            output_feature: Target values for training.

        Returns:
            GridSearchedBestModel: Named tuple containing the best model and its parameters.

        Raises:
            HousingException: If an error occurs during grid search.
        """
        try:
            # Instantiate GridSearchCV class
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name)

            # Initialize GridSearchCV with model and parameters
            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                  self.grid_search_property_data)

            # Log training start
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__} Started." {"<<"*30}'
            logging.info(message)
            # Perform grid search
            grid_search_cv.fit(input_feature, output_feature)
            # Log training completion
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__}" completed {"<<"*30}'
            # Store best model details
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                             model=initialized_model.model,
                                                             best_model=grid_search_cv.best_estimator_,
                                                             best_parameters=grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_)
            
            return grid_searched_best_model
        except Exception as e:
            raise HousingException(e, sys) from e

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """Initialize and return a list of model details from configuration.

        Returns:
            List[InitializedModelDetail]: List of initialized model details.

        Raises:
            HousingException: If an error occurs during model initialization.
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():
                # Get model configuration
                model_initialization_config = self.models_initialization_config[model_serial_number]
                # Load model class
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                            class_name=model_initialization_config[CLASS_KEY])
                model = model_obj_ref()
                
                # Update model parameters if provided
                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                 property_data=model_obj_property_data)

                # Get parameter grid for grid search
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"

                # Create initialized model detail
                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name)

                initialized_model_list.append(model_initialization_config)

            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise HousingException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                            input_feature,
                                                            output_feature) -> GridSearchedBestModel:
        """Perform parameter search for a single initialized model.

        Args:
            initialized_model (InitializedModelDetail): Model details including model and parameter grid.
            input_feature: Input features for training.
            output_feature: Target values for training.

        Returns:
            GridSearchedBestModel: Named tuple containing the best model and its parameters.

        Raises:
            HousingException: If an error occurs during parameter search.
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                     input_feature=input_feature,
                                                     output_feature=output_feature)
        except Exception as e:
            raise HousingException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self,
                                                             initialized_model_list: List[InitializedModelDetail],
                                                             input_feature,
                                                             output_feature) -> List[GridSearchedBestModel]:
        """Perform parameter search for a list of initialized models.

        Args:
            initialized_model_list (List[InitializedModelDetail]): List of model details.
            input_feature: Input features for training.
            output_feature: Target values for training.

        Returns:
            List[GridSearchedBestModel]: List of best models and their parameters.

        Raises:
            HousingException: If an error occurs during parameter search.
        """
        try:
            self.grid_searched_best_model_list = []
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model_list,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise HousingException(e, sys) from e

    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number: str) -> InitializedModelDetail:
        """Retrieve model details by serial number.

        Args:
            model_details (List[InitializedModelDetail]): List of model details.
            model_serial_number (str): Serial number of the model to retrieve.

        Returns:
            InitializedModelDetail: Details of the specified model.

        Raises:
            HousingException: If the model is not found or an error occurs.
        """
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise HousingException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                         base_accuracy=0.6) -> BestModel:
        """Select the best model from grid search results.

        Args:
            grid_searched_best_model_list (List[GridSearchedBestModel]): List of grid search results.
            base_accuracy (float, optional): Minimum acceptable accuracy. Defaults to 0.6.

        Returns:
            BestModel: Named tuple containing the best model and its parameters.

        Raises:
            HousingException: If no model meets the base accuracy or an error occurs.
        """
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found:{grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score

                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise HousingException(e, sys) from e

    def get_best_model(self, X, y, base_accuracy=0.6) -> BestModel:
        """Initialize models, perform grid search, and return the best model.

        Args:
            X: Input features for training.
            y: Target values for training.
            base_accuracy (float, optional): Minimum acceptable accuracy. Defaults to 0.6.

        Returns:
            BestModel: Named tuple containing the best model and its parameters.

        Raises:
            HousingException: If an error occurs during model selection.
        """
        try:
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                 base_accuracy=base_accuracy)
        except Exception as e:
            raise HousingException(e, sys)