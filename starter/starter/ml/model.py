import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
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

def train_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    """
    This function trains a given pipeline on the data passed as arguments
    """
    return model.fit(X_train, y_train)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pass
