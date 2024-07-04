'''
This script loads the dataframe after feature engineering and completes training
of the model. The model artifact is saved as a pickle file, and the metrics are
saved in a JSON file
'''

import json
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import yaml


with open('parameters.yaml', encoding='utf-8') as f:
    params = yaml.safe_load(f)

C = params['model']['parameters']['C']
penalty = params['model']['parameters']['penalty']


if __name__ == '__main__':
    df = pd.read_csv('data/eng_census.gz').convert_dtypes()
    y = df.pop('target__salary').rename('salary')
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )

    lr = LogisticRegression(C=C, penalty=penalty, max_iter=1000, solver='saga')
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    score = f1_score(y_test, y_pred)

    with open('model/score.json', 'w', encoding='utf-8') as f:
        json.dump({'f1_score': score}, f)

    with open('model/logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr, f)
