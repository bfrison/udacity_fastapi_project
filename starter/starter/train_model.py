'''
This script loads the dataframe after feature engineering and completes training
of the model. The model artifact is saved as a pickle file, and the metrics are
saved in a JSON file
'''

import json
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from ml.model import (
    compute_model_metrics,
    create_pipeline,
    score_strata,
    train_model,
)

with open('parameters.yaml', encoding='utf-8') as f:
    params = yaml.safe_load(f)

cat_cols = params['columns']['categorical']
num_cols = params['columns']['numerical']
strata_cols = params['columns']['strata']

C = params['model']['parameters']['C']
penalty = params['model']['parameters']['penalty']

if __name__ == '__main__':
    df_clean = pd.read_csv(os.path.join('data', 'clean_census.gz'))
    y = df_clean.pop('salary')

    # Optional enhancement, use K-fold cross validation instead of a train-test
    # split.
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean, y, test_size=0.2, random_state=42
    )

    pipeline = create_pipeline(cat_cols, num_cols, C, penalty)

    pipeline = train_model(pipeline, X_train, y_train)

    scores = compute_model_metrics(pipeline, X_test, y_test)

    strata_scores = score_strata(pipeline, X_test, y_test, strata_cols)

    with open(os.path.join('model', 'score.json'), 'w', encoding='utf-8') as f:
        json.dump(scores, f)

    with open(
        os.path.join('model', 'strata_score.json'), 'w', encoding='utf-8'
    ) as f:
        json.dump(strata_scores, f)

    with open(os.path.join('model', 'logistic_regression.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)
