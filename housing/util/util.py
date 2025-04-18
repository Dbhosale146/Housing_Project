import yaml
from housing.exception import HousingException
import os
import sys
import numpy as np
import dill
import pandas as pd
from housing.constant import *

# Constants from housing.constant (e.g., DATASET_SCHEMA_COLUMNS_KEY) are assumed to be defined


def write_yaml_file(file_path: str, data: dict = None):
    """Write a dictionary to a YAML file.

    Args:
        file_path (str): Path where the YAML file will be saved.
        data (dict, optional): Dictionary to write to the file. Defaults to None.

    Raises:
        HousingException: If an error occurs during file writing.
    """
    try:
        # Create the directory for the file if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Open and write to the YAML file
        with open(file_path, "w") as yaml_file:
            if data is not None:
                yaml.dump(data, yaml_file)
    except Exception as e:
        raise HousingException(e, sys)


def read_yaml_file(file_path: str) -> dict:
    """Read a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Contents of the YAML file.

    Raises:
        HousingException: If an error occurs during file reading.
    """
    try:
        # Open and read the YAML file
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HousingException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """Save a NumPy array to a file.

    Args:
        file_path (str): Path where the NumPy array will be saved.
        array (np.array): NumPy array to save.

    Raises:
        HousingException: If an error occurs during file saving.
    """
    try:
        # Create the directory for the file if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        # Save the NumPy array to the file
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise HousingException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """Load a NumPy array from a file.

    Args:
        file_path (str): Path to the file containing the NumPy array.

    Returns:
        np.array: Loaded NumPy array.

    Raises:
        HousingException: If an error occurs during file loading.
    """
    try:
        # Load the NumPy array from the file
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise HousingException(e, sys) from e


def save_object(file_path: str, obj):
    """Save a Python object to a file using dill serialization.

    Args:
        file_path (str): Path where the object will be saved.
        obj: Any Python object to serialize.

    Raises:
        HousingException: If an error occurs during object saving.
    """
    try:
        # Create the directory for the file if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        # Serialize and save the object using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise HousingException(e, sys) from e


def load_object(file_path: str):
    """Load a Python object from a file using dill deserialization.

    Args:
        file_path (str): Path to the file containing the serialized object.

    Returns:
        Any: Deserialized Python object.

    Raises:
        HousingException: If an error occurs during object loading.
    """
    try:
        # Deserialize and load the object using dill
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise HousingException(e, sys) from e


def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    """Load a CSV dataset and validate its columns against a schema.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        schema_file_path (str): Path to the YAML file containing the dataset schema.

    Returns:
        pd.DataFrame: Validated DataFrame with columns matching the schema.

    Raises:
        HousingException: If the dataset columns do not match the schema or an error occurs.
    """
    try:
        # Read the schema from the YAML file
        dataset_schema = read_yaml_file(schema_file_path)
        schema = dataset_schema[DATASET_SCHEMA_COLUMNS_KEY]

        # Load the CSV dataset into a DataFrame
        dataframe = pd.read_csv(file_path)

        # Initialize error message for schema validation
        error_message = ""

        # Validate each column in the DataFrame against the schema
        for column in dataframe.columns:
            if column in list(schema.keys()):
                # Convert column to the specified type in the schema
                dataframe[column].astype(schema[column])
            else:
                # Append error message for missing columns
                error_message = f"{error_message} \nColumn: [{column}] is not in the schema."
        if len(error_message) > 0:
            raise Exception(error_message)
        return dataframe
    except Exception as e:
        raise HousingException(e, sys) from e