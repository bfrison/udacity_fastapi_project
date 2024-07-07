'''
This utility module contains various functions related to the machine learning
model
'''

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml


with open('parameters.yaml', encoding='utf-8') as f:
    params = yaml.safe_load(f)

cat_cols = params['columns']['categorical']
num_cols = params['columns']['numerical']

C = params['model']['parameters']['C']
penalty = params['model']['parameters']['penalty']


def create_pipeline(
    cat_cols: list, num_cols: list, C: float, penalty: str
) -> Pipeline:
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


def train_model(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame
) -> Pipeline:
    '''
    This function trains a given pipeline on the data passed as arguments
    '''
    return model.fit(X_train, y_train)


def compute_model_metrics(
    model: Pipeline, X: pd.DataFrame, y_true: pd.Series, pos_label: str = '>50K'
) -> dict:
    '''
    This function uses f1_scoring, precision and recall to score a given
    pipeline on the data passed as arguments
    '''
    y_pred = inference(model, X)

    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=1)
    precision = precision_score(
        y_true, y_pred, pos_label=pos_label, zero_division=1
    )
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=1)

    scores = {
        'f1_score': f1,
        'precision_score': precision,
        'recall_score': recall,
    }
    return scores


def inference(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    '''
    This function infers results from the model and data passed as arguments
    '''
    y_pred = pd.Series(model.predict(X), index=X.index, name='salary_pred')

    return y_pred


def score_strata(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, column: str
) -> dict:
    '''
    This function scores stratifies the data based on the columns passed as an
    argument and scores each stratum separately
    '''
    scores_dict = {}
    for val, idxs in X.groupby(column).groups.items():
        X_slice = X.loc[idxs]
        y_slice = y.loc[idxs]
        scores = compute_model_metrics(model, X_slice, y_slice)
        scores_dict[val] = scores

    return scores_dict
