'''
This script infers results from a model and data
'''

import pickle

import os
import pandas as pd

from train import infer


if __name__ == '__main__':
    model_path = os.path.join('model', 'logistic_regression.pkl')
    data_path = os.path.join('data', 'clean_census.gz')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    df = pd.read_csv(data_path).tail()

    y_pred = infer(model, df)

    print(y_pred.to_json(orient='records'))
