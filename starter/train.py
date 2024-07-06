'''
This script loads the dataframe after feature engineering and completes training
of the model. The model artifact is saved as a pickle file, and the metrics are
saved in a JSON file
'''

import json
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml


with open('parameters.yaml', encoding='utf-8') as f:
    params = yaml.safe_load(f)

cat_cols = params['columns']['categorical']
num_cols = params['columns']['numerical']

C = params['model']['parameters']['C']
penalty = params['model']['parameters']['penalty']


def create_pipeline(cat_cols, num_cols, C, penalty):
    '''
    This function returns the pipeline that will be applied on preprocessed data
    '''
    col_transf = ColumnTransformer(
        [
            (
                'one_hot_encoder',
                make_pipeline(
                    SimpleImputer(
                        missing_values=pd.NA,
                        strategy='constant',
                        fill_value='Unknown',
                    ),
                    OneHotEncoder(sparse_output=False),
                ),
                cat_cols,
            ),
            ('standard_scaler', StandardScaler(), num_cols),
        ]
    )

    lr = LogisticRegression(C=C, penalty=penalty, max_iter=1000, solver='saga')

    pipeline = Pipeline(
        [('column_transformer', col_transf), ('logistic_regressor', lr)]
    )

    return pipeline


if __name__ == '__main__':
    df = pd.read_csv('data/clean_census.gz').convert_dtypes()
    y = df.pop('salary')
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )

    pipeline = create_pipeline(cat_cols, num_cols, C, penalty)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    score = f1_score(y_test, y_pred, pos_label='>50K')

    with open('model/score.json', 'w', encoding='utf-8') as f:
        json.dump({'f1_score': score}, f)

    with open('model/logistic_regression.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
