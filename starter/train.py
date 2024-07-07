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


# def create_pipeline(
#     cat_cols: list, num_cols: list, C: float, penalty: str
# ) -> Pipeline:
#     '''
#     This function returns the pipeline that will be applied on preprocessed data
#     '''
#     col_transf = ColumnTransformer(
#         [
#             (
#                 'one_hot_encoder',
#                 make_pipeline(
#                     SimpleImputer(
#                         missing_values=pd.NA,
#                         strategy='constant',
#                         fill_value='Unknown',
#                     ),
#                     OneHotEncoder(sparse_output=False),
#                 ),
#                 cat_cols,
#             ),
#             ('standard_scaler', StandardScaler(), num_cols),
#         ]
#     )
# 
#     lr = LogisticRegression(C=C, penalty=penalty, max_iter=1000, solver='saga')
# 
#     pipeline = Pipeline(
#         [('column_transformer', col_transf), ('logistic_regressor', lr)]
#     )
# 
#     return pipeline
# 
# 
# def train(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
#     '''
#     This function trains a given pipeline on the data passed as arguments
#     '''
#     return model.fit(X, y)


def infer(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    '''
    This function infers results from the model and data passed as arguments
    '''
    y_pred = pd.Series(model.predict(X), index=X.index, name='salary_pred')

    return y_pred


def score(
    model: Pipeline, X: pd.DataFrame, y_true: pd.Series, pos_label: str = '>50K'
) -> float:
    '''
    This function uses f1_scoring to score a given pipeline on the data passed
    as arguments
    '''
    return f1_score(y_true, infer(model, X), pos_label=pos_label)


def score_strata(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, column: str
) -> pd.Series:
    '''
    This function scores stratifies the data based on the columns passed as an
    argument and scores each stratum separately
    '''
    scores_dict = {}
    for val, idxs in X.groupby(column).groups.items():
        X_slice = X.loc[idxs]
        y_slice = y.loc[idxs]
        f1_score_val = score(model, X_slice, y_slice)
        scores_dict[val] = f1_score_val

    return pd.Series(scores_dict)


if __name__ == '__main__':
    df = pd.read_csv('data/clean_census.gz').convert_dtypes()
    y = df.pop('salary')
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )

    pipeline = create_pipeline(cat_cols, num_cols, C, penalty)

    pipeline = train(pipeline, X_train, y_train)

    f1_score_val = score(pipeline, X_test, y_test)

    with open('model/score.json', 'w', encoding='utf-8') as f:
        json.dump({'f1_score': f1_score_val}, f)

    with open('model/logistic_regression.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
