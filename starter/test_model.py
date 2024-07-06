import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import yaml

from infer import infer
from preprocessing import preprocessing
from train import create_pipeline, score, train

@pytest.fixture
def data_dir():
    return 'data'

@pytest.fixture
def input_file():
    return 'census.csv'

@pytest.fixture
def df_clean(data_dir, input_file):
    return preprocessing(data_dir, input_file).head(10)

@pytest.fixture
def salary(df_clean):
    return df_clean.salary.head(10)

@pytest.fixture
def parameters():
    with open('parameters.yaml', encoding='utf-8') as f:
        params = yaml.safe_load(f)

    return params

@pytest.fixture
def pipeline(parameters):
    cat_cols = parameters['columns']['categorical']
    num_cols = parameters['columns']['numerical']

    C = parameters['model']['parameters']['C']
    penalty = parameters['model']['parameters']['penalty']

    pipeline = create_pipeline(cat_cols, num_cols, C, penalty)

    return pipeline

@pytest.fixture
def trained_pipeline(pipeline, df_clean, salary):
    pipeline = train(pipeline, df_clean, salary)

    return pipeline

def test_preprocessing_dataframe(df_clean):

    assert isinstance(df_clean, pd.DataFrame), 'Data is not in the Pandas DataFrame format'
    assert len(df_clean) > 0, 'Data contains now rows'
    assert 'salary' in df_clean, 'Data does not containt the \'salary\' target column'

def test_preprocessing_deduplication(df_clean):

    num_duplicates = df_clean.duplicated().sum()

    assert num_duplicates == 0, f'There are {num_duplicates:,d} duplicates in data'

def test_create_pipeline(pipeline):

    assert isinstance(pipeline, Pipeline), 'Model is not an instance of Sci-Kit Learn Pipeline'
    assert isinstance(pipeline[-1], LogisticRegression), 'Model\'s regressor is not an instance of LogisticRegressor'

def test_train(trained_pipeline):

    assert trained_pipeline.__sklearn_is_fitted__(), 'Model is not fitted after training'

def test_score(trained_pipeline, df_clean, salary):

    f1_score_val = score(trained_pipeline, df_clean, salary)

    assert isinstance(f1_score_val, float), 'F1 score is not a float'

def test_infer(trained_pipeline, df_clean, salary):

    y_pred = infer(trained_pipeline, df_clean)

    assert isinstance(y_pred, pd.Series), 'Inferred data is not in the Pandas Series format'
    assert set(y_pred) == set(salary), 'The set of inferred values does not correspond to original target data'
