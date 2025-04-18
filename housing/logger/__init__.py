import logging
from datetime import datetime
import os
import pandas as pd
from housing.constant import get_current_time_stamp

# Define the directory for storing log files
LOG_DIR = "logs"


def get_log_file_name():
    """Generate a log file name with a timestamp.

    Returns:
        str: Log file name in the format 'log_<timestamp>.log'.
    """
    return f"log_{get_current_time_stamp()}.log"


# Generate the log file name
LOG_FILE_NAME = get_log_file_name()

# Create the logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Configure the logging system
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w",  # Overwrite the log file each time
    format='[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s',
    level=logging.INFO  # Set logging level to INFO
)


def get_log_dataframe(file_path):
    """Convert a log file into a pandas DataFrame.

    Args:
        file_path (str): Path to the log file.

    Returns:
        pd.DataFrame: DataFrame containing formatted log messages with a single 'log_message' column.
    """
    # Initialize an empty list to store log data
    data = []
    # Read the log file line by line
    with open(file_path) as log_file:
        for line in log_file.readlines():
            # Split each line by the delimiter '^;'
            data.append(line.split("^;"))

    # Create a DataFrame from the log data
    log_df = pd.DataFrame(data)
    # Define column names for the DataFrame
    columns = ["Time stamp", "Log Level", "line number", "file name", "function name", "message"]
    log_df.columns = columns

    # Create a new column combining timestamp and message
    log_df["log_message"] = log_df['Time stamp'].astype(str) + ":$" + log_df["message"]

    # Return a DataFrame with only the 'log_message' column
    return log_df[["log_message"]]