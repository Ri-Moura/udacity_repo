'''
Library created to import data, pre-process, perform feature engineering, train and 
test model and plot evaluation metrics.

Author: Ricardo Moura
Date: January, 2023
'''


# import libraries
import os
import logging

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

logging.basicConfig(
    filename='./test_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

os.environ['QT_QPA_PLATFORM']='offscreen'

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    
    df = pd.read_csv(pth)
    
    return df

        
def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20,10)) 
    df['Churn'].hist()
    plt.plot()
    plt.savefig('../images/churn_histogram.png')

    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist()
    plt.plot()
    plt.savefig('../images/age_histogram.png')

    plt.figure(figsize=(20,10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('../images/marital_status_countplot.png')

    plt.figure(figsize=(20,10)) 
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('../images/totaltrans_histogram.png')

    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig("../images/corr_plot.png")
    
def _find_category_churn(df, category_name, response):
    '''
    Private function for finding a category churn.
    Args:
        churn_dataframe: (pandas.core.frame.DataFrame) pandas dataframe with the 'churn'
        column and category information.
        category_name: (str) a string describing the dataframe column
        to be encoded.
    Returns:
        result: (dict) a dictionary with each category and propotion of
        churn in that category as key:value pair.
    '''

    churn_info = df.groupby(category_name)[response].mean()
    result = dict(zip(churn_info.index, churn_info.values))

    return result


def encoder_helper(df, category_lst, response):
    '''
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category.
    Args:
        df: (pandas.core.frame.DataFrame) pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be
        used for naming variables or index y column.
    Returns:
        df: (pandas.core.frame.DataFrame) pandas dataframe with encoded columns
    '''

    for category in category_lst:
        mapping_information = _find_category_churn(df, category, response)
        df[category + '_churn'] = df[category].map(mapping_information)

    return df

    
    
def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    keep_cols = ['Customer_Age', 'Dependent_count',
                 'Months_on_book', 'Total_Relationship_Count',
                 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 
                 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn',
                 'Marital_Status_Churn', 
                 'Income_Category_Churn', 'Card_Category_Churn']

    dataframe_encoded = encoder_helper(df,
                                       ['Gender',
                                        'Education_Level',
                                        'Marital_Status',
                                        'Income_Category',
                                        'Card_Category'],
                                       response)

    X = dataframe_encoded[keep_cols]
    y = dataframe_encoded[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.figure(figsize=(12, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rf_results')

    plt.figure(figsize=(12, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/logistic_results')
    


def feature_importance_plot(model, X_data, output_pth="../images/featureimportance_plot.png"):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    
    # Create plot
    plt.figure(figsize=(20,5))
    
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);
    plt.savefig(output_pth)

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    
    # fitting models
    cv_rfc = GridSearchCV(estimator=rfc, 
                          param_grid=param_grid, 
                          cv=5)
    
    cv_rfc.fit(X_train, y_train)
    
    lrc.fit(X_train, y_train)
    
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    
    # generating preds
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    
    feature_importance_plot(
        cv_rfc,
        X_train,
        './images/results/feature_importances')

if __name__ == "__main__":
    FILE_PATH = sys.argv[1]

    start = time.time()

    print(f'>> STARTING AT {start}')
    print('>> IMPORTING DATA')
    DATA = import_data(FILE_PATH)

    print('>> PERFORMING EDA')
    perform_eda(DATA)

    print('>> PERFORMING FEATURE ENGINEERING')
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DATA, 'churn')

    print('>> TRAINING MODELS')
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

    print('>> DONE')

    print('Duration: {} seconds'.format(time.time() - start))