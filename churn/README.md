# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The project has the objective to pre-process data, process some feature engineerings, train and test
random forest and logistic regression models to predict costumer churn in a bank.

## Files and data description
.
├── data <-- raw data
├── models <-- trained and serialized models.
├── notebooks <-- Jupyter notebooks.
├── images <-- figures generate on this project.
├── src <-- source codes
├── logs <-- logs from test code
├── README.md <-- project documentation.
└── requirements.txt <-- project dependencies.

## Running Files
To run this project first you should create and activate an environment.

Using pyenv:

pyenv install 3.8.12

pyenv virtualenv 3.8.12 <ENV_NAME>

pyenv global <ENV_NAME>

Use pip to install the necessary requirements.

pip install -r requirements.txt

Now you should be able to run the whole project

Use:

python churn_library.py

To run the library and

python churn_script_logging_and_tests.py

To run the tests and log and the information about the tests.