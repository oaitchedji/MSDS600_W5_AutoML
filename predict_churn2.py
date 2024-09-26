import pandas as pd 
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    """ Loads churn data into a Dataframe from a string filepath."""
    df = read_csv (filepath, index_col= 'CustomerID')
    return df


def make_predictions(df):
    """ Uses the pycaret best model to make predictions on data in the df dataframe"""
    model = load_model(GBC)
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'Churn_prediction'}), axis=1, inplace=True
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No churn'}),
    inplace=True
    return predictions['Churn_prediction']

if __name__ == "__main__":
    df = load_data('/Users/dandelion/Library/CloudStorage/OneDrive-RegisUniversity/MSHI/MSDS 600 Intro to Data Science/MSDS 600 NBC/W5 - Automated Data Science/new_churn_data.csv')
    predictions = make_predictions (df)
    print('predictions:')
    print(predictions)