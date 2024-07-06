import json
import pickle

import os
import pandas as pd

def infer(model_path: str, X_json: str) -> pd.Series:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X = pd.DataFrame.from_records(json.loads(X_json))

    y_pred = pd.Series(model.predict(X), index=X.index, name='salary_pred')

    return y_pred

if __name__ == '__main__':
    model_path = os.path.join('model', 'logistic_regression.pkl')
    data_path = os.path.join('data', 'clean_census.gz')
    df = pd.read_csv(data_path).tail()

    y_pred = infer(model_path, df.to_json(orient='records'))

    print(y_pred.to_json(orient='records'))
