# Predict Customer Churn

## Project Overview

The **Predict Customer Churn** project is part of the ML DevOps Engineer Nanodegree program offered by Udacity. The goal of the project is to pre-process data, perform feature engineering, train and test a random forest and logistic regression model to predict customer churn in a bank.

## File and Data Description
The project includes the following files and directories:
- `data`: contains raw data for the project
- `models`: contains trained and serialized models
- `notebooks`: contains Jupyter notebooks used for data analysis and model development
- `images`: contains figures generated during the project
- `src`: contains source code for the project
- `logs`: contains logs from test code
- `README.md`: project documentation
- `requirements.txt`: lists project dependencies

## Running the Project

To run this project, you will need to create and activate an environment using `pyenv`:

1. Install Python 3.8.12: `pyenv install 3.8.12`
2. Create a virtual environment: `pyenv virtualenv 3.8.12 <ENV_NAME>`
3. Activate the virtual environment: `pyenv global <ENV_NAME>`
4. Install project dependencies: `pip install -r requirements.txt`

Once the environment is set up and the dependencies are installed, you can run the project:

1. To run the library, use: `python churn_library.py`
2. To run the tests and log the results, use: `python churn_script_logging_and_tests.py`