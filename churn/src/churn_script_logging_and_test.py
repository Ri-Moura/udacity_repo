'''
Library created to perform some test and log informations about tests

Author: Ricardo Moura
Date: January, 2023
'''

# import libraries
import os
import logging
import pandas as pd
import pytest
import churn_project.src.churn_library as clb


logging.basicConfig(
    filename='./test_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

os.environ['QT_QPA_PLATFORM']='offscreen'

@pytest.fixture(name='data_path')
def fixture_data_path():
    """
    fixture for path of test data
    """
    return 'data/bank_data.csv'


@pytest.fixture(name='data_not_encoded')
def fixture_data_not_encoded(data_path):
    """
    fixture for returning data that was not encoded yet.
    Args:
        data_path: path for test data
    Returns:
        imported_data: data imported using the `import_data` function.
    """

    imported_data = clb.import_data(data_path)

    return imported_data


@pytest.fixture(name='eda_file_names')
def fixture_eda_file_names():
    """
    fixture for returning names of the files that must be tested.
    """

    return [
        'churn_distribution.png',
        'customer_age_distribution.png',
        'marital_status_distribution.png',
        'total_transaction_distribution.png',
        'heatmap.png']


@pytest.fixture(name='result_file_names')
def fixture_result_file_names():
    """
    fixture for returning names of files that must be tested.
    """
    return ['feature_importances.png', 'logistic_results.png',
            'rf_results.png', 'roc_curve_result.png']


@pytest.fixture(name='columns_to_encode')
def fixture_columns_to_encode():
    """
    fixture for returning column names that must be encoded
    """
    return ['Gender', 'Education_Level',
            'Marital_Status', 'Income_Category',
            'Card_Category']


@pytest.fixture(name='response_variable_name')
def fixture_response_variable_name():
    """
    fixture for returning name of the response variable
    """
    return 'churn'


@pytest.fixture(name='model_features')
def fixture_model_features():
    """
    fixture for returning name of the features that must be kept
    """
    return [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_churn',
        'Education_Level_churn',
        'Marital_Status_churn',
        'Income_Category_churn',
        'Card_Category_churn']


def test_import(data_path):
    '''
    test data import - this example is completed for you to assist with the
    other test functions
    '''
    try:
        data_frame = clb.import_data(data_path)
        logging.info("Testing import_data: SUCCESS")
    except (FileNotFoundError, ValueError) as err:
        logging.error(
            "Testing import_eda: The file wasn't found or type was incorrect.")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to \
                have rows and columns")
        raise err


def test_eda(eda_file_names):
    '''
    test perform eda function
    '''

    for file in eda_file_names:
        try:
            assert os.path.exists('images/eda/' + file)
            logging.info("Testing EDA: SUCCESS (%s)",file)
        except BaseException as err:
            logging.error(
                "Testing EDA: FAILED - the file %s was not found", file)
            raise FileNotFoundError('The %s was not found' % file) from err


def test_encoder_helper(
        data_not_encoded,
        columns_to_encode,
        response_variable_name):
    '''
    test encoder helper
    '''

    data_encoded = clb.encoder_helper(
        data_not_encoded,
        columns_to_encode,
        response_variable_name
    )

    for encoded_column in columns_to_encode:
        try:
            assert encoded_column + '_churn' in data_encoded
            logging.info("Testing ENCODER HELPER: SUCCESS")
        except BaseException as err:
            logging.error(
                "Testing ENCODER HELPER: \
                    FAILED - %s_churn was not found.", encoded_column)
            raise KeyError(
                'The %s_churn was not found.' % encoded_column) from err


def test_perform_feature_engineering(
        data_not_encoded,
        response_variable_name,
        model_features):
    '''
    test perform_feature_engineering
    '''

    model_training_data = clb.perform_feature_engineering(
        data_not_encoded, response_variable_name)

    # testing if the function returns all four datasets
    try:
        assert len(model_training_data) == 4
        logging.info(
            "Testing PERFORM FEATURE ENGINEERING: \
                SUCCESS (returns all four datasets)")
    except BaseException as err:
        logging.error(
            "Testing PERFORM FEATURE ENGINEERING: \
                SUCCESS (returns all four datasets)")
        raise FileNotFoundError('The function must return all four datasets.') from err

    # testing if X_train, X_test are the correct type
    for dataset in (model_training_data[0], model_training_data[1]):
        try:
            assert isinstance(dataset, pd.core.frame.DataFrame)
            logging.info(
                "Testing PERFORM FEATURE ENGINEERING: \
                    SUCCESS (X_train, X_test have the correct types)")
        except BaseException as err:
            logging.error(
                "Testing PERFORM FEATURE ENGINEERING: \
                    FAILED (X is not in correct type.)")
            raise TypeError(
                'X_train and X_test must be \
                    instances of a pandas Dataframe.') from err

    # testing if y_train, y_test are the correct type
    for dataset in (model_training_data[2], model_training_data[3]):
        try:
            assert isinstance(dataset, pd.core.series.Series)
            logging.info(
                "Testing PERFORM FEATURE ENGINEERING:\
                     SUCCESS (y_train, y_test have the correct types)")
        except BaseException as err:
            logging.error(
                "Testing PERFORM FEATURE ENGINEERING:\
                     FAILED (y is not in correct type.)")
            raise TypeError(
                'y_train and y_test must be instances of a pandas Series.') from err

    # testing if X_train and X_test have all the necessary columns
    for feature in model_features:
        try:
            assert feature in model_training_data[0]
            logging.info(
                "Testing PERFORM FEATURE ENGINEERING: SUCCESS \
                    (feature %s is present in X_train data)", feature)
        except BaseException as err:
            logging.error(
                "Testing PERFORM FEATURE ENGINEERING: FAILED \
                    (feature %s is present in X_train data)", feature)
            raise KeyError('The %s was not found in X_train.' % feature) from err

        try:
            assert feature in model_training_data[1]
            logging.info(
                "Testing PERFORM FEATURE ENGINEERING: SUCCESS \
                    (feature %s is present in X_test data)", feature)
        except BaseException as err:
            logging.error(
                "Testing PERFORM FEATURE ENGINEERING: FAILED \
                    (feature %s is not present in X_test data)", feature)
            raise KeyError('The %s was not found in X_test.' % feature) from err


def test_train_models(
        data_not_encoded,
        result_file_names,
        response_variable_name):
    '''
    test train_models
    '''

    # performing the model training
    features_x_train, features_x_test, y_train, y_test = clb.perform_feature_engineering(
        data_not_encoded, response_variable_name)
    clb.train_models(features_x_train, features_x_test, y_train, y_test)

    # checking for models saved
    local_path = os.listdir('models/')

    try:
        assert len(local_path) >= 2
        logging.info("Testing TRAIN MODELS: SUCCESS (number of models)")
    except BaseException as err:
        logging.error(
            "Testing TRAIN MODELS: FAILED - %s were trained.", len(local_path))
        raise FileNotFoundError(
            'At least two models must be trained and saved.') from err

    # checking result images
    for file in result_file_names:
        try:
            assert os.path.exists('images/results/' + file)
            logging.info("Testing TRAIN MODEL: SUCCESS")
        except BaseException as err:
            logging.error(
                "Testing TRAIN MODEL: FAILED - the file %s \
                    was not found", file)
            raise FileNotFoundError('The %s was not found' % file) from err


if __name__ == "__main__":

    PATH = 'data/bank_data.csv'

    # testing import
    test_import(PATH)

    # testing eda
    test_eda(['churn_distribution.png',
              'customer_age_distribution.png',
              'marital_status_distribution.png',
              'total_transaction_distribution.png',
              'heatmap.png'])

    # testing encoder
    DATA = clb.import_data(PATH)

    test_encoder_helper(
        data_not_encoded=DATA,
        columns_to_encode=[
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'],
        response_variable_name='churn')

    # testing feature_eng
    test_perform_feature_engineering(
        data_not_encoded=DATA,
        response_variable_name='churn',
        model_features=[
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Gender_churn',
            'Education_Level_churn',
            'Marital_Status_churn',
            'Income_Category_churn',
            'Card_Category_churn'])

    # testing train_models
    test_train_models(
        data_not_encoded=DATA,
        result_file_names=[
            'feature_importances.png',
            'logistic_results.png',
            'rf_results.png',
            'roc_curve_result.png'],
        response_variable_name='churn')