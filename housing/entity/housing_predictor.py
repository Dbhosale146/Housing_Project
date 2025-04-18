import os
import sys

from housing.exception import HousingException
from housing.util.util import load_object

import pandas as pd


class HousingData:
    """
    HousingData is a structured representation of input data for the model. 
    It holds all features necessary to predict the housing price in California.
    """

    def __init__(self,
                 longitude: float,
                 latitude: float,
                 housing_median_age: float,
                 total_rooms: float,
                 total_bedrooms: float,
                 population: float,
                 households: float,
                 median_income: float,
                 ocean_proximity: str,
                 median_house_value: float = None
                 ):
        """
        Initializes the housing data attributes.
        """
        try:
            self.longitude = longitude
            self.latitude = latitude
            self.housing_median_age = housing_median_age
            self.total_rooms = total_rooms
            self.total_bedrooms = total_bedrooms
            self.population = population
            self.households = households
            self.median_income = median_income
            self.ocean_proximity = ocean_proximity
            self.median_house_value = median_house_value
        except Exception as e:
            raise HousingException(e, sys) from e

    def get_housing_input_data_frame(self):
        """
        Converts the housing data attributes into a pandas DataFrame 
        suitable for model prediction.
        """
        try:
            housing_input_dict = self.get_housing_data_as_dict()
            return pd.DataFrame(housing_input_dict)
        except Exception as e:
            raise HousingException(e, sys) from e

    def get_housing_data_as_dict(self):
        """
        Returns housing data in dictionary format.
        """
        try:
            input_data = {
                "longitude": [self.longitude],
                "latitude": [self.latitude],
                "housing_median_age": [self.housing_median_age],
                "total_rooms": [self.total_rooms],
                "total_bedrooms": [self.total_bedrooms],
                "population": [self.population],
                "households": [self.households],
                "median_income": [self.median_income],
                "ocean_proximity": [self.ocean_proximity]}
            return input_data
        except Exception as e:
            raise HousingException(e, sys)


class HousingPredictor:
    """
    HousingPredictor loads the latest trained model and performs predictions
    on the input DataFrame.
    """

    def __init__(self, model_dir: str):
        """
        Initializes the predictor with the given model directory path.
        """
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise HousingException(e, sys) from e

    def get_latest_model_path(self):
        """
        Finds and returns the path of the latest model (.pkl file) in the model directory.
        """
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                raise Exception(f"Model directory not found: {self.model_dir}")

            # Filter only versioned folders (digits only)
            folder_names = [int(name) for name in os.listdir(self.model_dir) if name.isdigit()]

            if not folder_names:
                raise Exception(f"No versioned model folders found in: {self.model_dir}")

            latest_model_dir = os.path.join(self.model_dir, str(max(folder_names)))

            # Get .pkl file (usually named 'model.pkl' or 'pipeline.pkl')
            model_files = [f for f in os.listdir(latest_model_dir) if f.endswith(".pkl")]

            if not model_files:
                raise Exception(f"No .pkl model found in latest directory: {latest_model_dir}")

            latest_model_path = os.path.join(latest_model_dir, model_files[0])
            return latest_model_path

        except Exception as e:
            raise HousingException(e, sys) from e

    def predict(self, X):
        """
        Predicts the median house value using the latest model and given input features X.
        """
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            median_house_value = model.predict(X)
            return median_house_value
        except Exception as e:
            raise HousingException(e, sys) from e
