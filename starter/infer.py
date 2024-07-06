'''
This script infers results from a model and data
'''

import pickle

import os
import pandas as pd
from sklearn.pipeline import Pipeline


def infer(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    '''
    This function infers results from the model and data passed as arguments
    '''
    y_pred = pd.Series(model.predict(X), index=X.index, name='salary_pred')

    return y_pred


if __name__ == '__main__':
    model_path = os.path.join('model', 'logistic_regression.pkl')
    data_path = os.path.join('data', 'clean_census.gz')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    df = pd.read_csv(data_path).tail()

    y_pred = infer(model, df)

    print(y_pred.to_json(orient='records'))
