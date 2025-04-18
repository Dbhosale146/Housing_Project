from flask import Flask, request, send_file, abort, render_template
import sys
import pip
from housing.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from housing.logger import logging, get_log_dataframe
from housing.exception import HousingException
import os
import json
from housing.config.configuration import Configuartion
from housing.constant import CONFIG_DIR, get_current_time_stamp
from housing.pipeline.pipeline import Pipeline
from housing.entity.housing_predictor import HousingPredictor, HousingData

# Define root directory and folder/file paths
ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "housing"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

# Define constants for context keys
HOUSING_DATA_KEY = "housing_data"
MEDIAN_HOUSING_VALUE_KEY = "median_house_value"

# Initialize Flask application
app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'housing'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    """Render the contents of the artifact directory or serve a file.

    Args:
        req_path (str): Path to the artifact directory or file relative to the root.

    Returns:
        str or flask.Response: HTML content, file, or rendered template with directory contents.

    Raises:
        flask.Abort: If the requested path does not exist (404).
    """
    # Create the housing directory if it doesn't exist
    os.makedirs("housing", exist_ok=True)
    # Log the requested path
    print(f"req_path: {req_path}")
    # Construct the absolute path
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if the path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Serve the file if it exists
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            # Read and return HTML file content
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # List files containing "artifact" in their path
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path)
             if "artifact" in os.path.join(abs_path, file_name)}

    # Prepare context for rendering directory contents
    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the home page.

    Returns:
        str: Rendered index.html template or error message if an exception occurs.
    """
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    """Render the experiment history page.

    Returns:
        str: Rendered experiment_history.html template with experiment data.
    """
    # Get experiment status DataFrame
    experiment_df = Pipeline.get_experiments_status()
    # Prepare context with experiment data as HTML table
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    """Start or check the status of the training pipeline.

    Returns:
        str: Rendered train.html template with training status and experiment data.
    """
    message = ""
    # Initialize pipeline with configuration
    pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
    # Check if pipeline is already running
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    # Prepare context with experiment data and message
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle housing price prediction requests.

    Returns:
        str: Rendered predict.html template with prediction results (POST) or input form (GET).
    """
    # Initialize default context
    context = {
        HOUSING_DATA_KEY: None,
        MEDIAN_HOUSING_VALUE_KEY: None
    }

    if request.method == 'POST':
        # Extract form data
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        housing_median_age = float(request.form['housing_median_age'])
        total_rooms = float(request.form['total_rooms'])
        total_bedrooms = float(request.form['total_bedrooms'])
        population = float(request.form['population'])
        households = float(request.form['households'])
        median_income = float(request.form['median_income'])
        ocean_proximity = request.form['ocean_proximity']

        # Create HousingData instance
        housing_data = HousingData(longitude=longitude,
                                  latitude=latitude,
                                  housing_median_age=housing_median_age,
                                  total_rooms=total_rooms,
                                  total_bedrooms=total_bedrooms,
                                  population=population,
                                  households=households,
                                  median_income=median_income,
                                  ocean_proximity=ocean_proximity)
        # Convert to DataFrame
        housing_df = housing_data.get_housing_input_data_frame()
        # Initialize predictor
        housing_predictor = HousingPredictor(model_dir=MODEL_DIR)
        # Make prediction
        median_housing_value = housing_predictor.predict(X=housing_df)
        # Update context with input data and prediction
        context = {
            HOUSING_DATA_KEY: housing_data.get_housing_data_as_dict(),
            MEDIAN_HOUSING_VALUE_KEY: median_housing_value
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    """Render the contents of the saved models directory or serve a file.

    Args:
        req_path (str): Path to the saved models directory or file relative to the root.

    Returns:
        flask.Response: File or rendered template with directory contents.

    Raises:
        flask.Abort: If the requested path does not exist (404).
    """
    # Create the saved_models directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    # Log the requested path
    print(f"req_path: {req_path}")
    # Construct the absolute path
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if the path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Serve the file if it exists
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # List all files in the directory
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    # Prepare context for rendering directory contents
    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    """Update or display the model configuration YAML file.

    Returns:
        str: Rendered update_model.html template with current model configuration.

    Raises:
        Exception: If an error occurs during YAML file operations (logged and returned as string).
    """
    try:
        if request.method == 'POST':
            # Get and parse new model configuration from form
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)
            # Write updated configuration to YAML file
            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        # Read current model configuration
        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        # Render template with configuration
        return render_template('update_model.html', result={"model_config": model_config})
    except Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    """Render the contents of the logs directory or display a log file as a DataFrame.

    Args:
        req_path (str): Path to the logs directory or file relative to the root.

    Returns:
        str: Rendered template with log file contents or directory listing.

    Raises:
        flask.Abort: If the requested path does not exist (404).
    """
    # Create the logs directory if it doesn't exist
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Log the requested path
    logging.info(f"req_path: {req_path}")
    # Construct the absolute path
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if the path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Display log file contents as a DataFrame
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # List all files in the directory
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    # Prepare context for rendering directory contents
    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    """Run the Flask application."""
    app.run()