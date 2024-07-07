import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import yaml

from preprocessing import preprocessing
from starter.ml.model import compute_model_metrics, create_pipeline, inference, train_model
from train import score_strata

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
    pipeline = train_model(pipeline, df_clean, salary)

    return pipeline

def test_preprocessing_dataframe(df_clean):

    assert isinstance(df_clean, pd.DataFrame), 'Data is not in the Pandas DataFrame format'
    assert len(df_clean) > 0, 'Data contains now rows'
    assert 'salary' in df_clean, 'Data does not contain the \'salary\' target column'

def test_preprocessing_deduplication(df_clean):

    num_duplicates = df_clean.duplicated().sum()

    assert num_duplicates == 0, f'There are {num_duplicates:,d} duplicates in data'

def test_create_pipeline(pipeline):

    assert isinstance(pipeline, Pipeline), 'Model is not an instance of Sci-Kit Learn Pipeline'
    assert isinstance(pipeline[-1], LogisticRegression), 'Model\'s regressor is not an instance of LogisticRegressor'

def test_train(trained_pipeline):

    assert trained_pipeline.__sklearn_is_fitted__(), 'Model is not fitted after training'

@pytest.mark.parametrize('score', ['f1_score', 'precision_score', 'recall_score']) 
def test_compute_model_metrics(trained_pipeline, df_clean, salary, score):

    score_vals = compute_model_metrics(trained_pipeline, df_clean, salary)

    assert f'{score}' in score_vals, f'{score} not in score values'
    assert isinstance(score_vals[f'{score}'], float), f'{score} is not a float'

def test_inference(trained_pipeline, df_clean, salary):

    y_pred = inference(trained_pipeline, df_clean)

    assert isinstance(y_pred, pd.Series), 'Inferred data is not in the Pandas Series format'
    assert set(y_pred) == set(salary), 'The set of inferred values does not correspond to original target data'

def test_score_strata(trained_pipeline, df_clean, salary):

    scores = score_strata(trained_pipeline, df_clean, salary, 'sex')

    assert isinstance(scores, pd.Series), 'Scores are not in the Pandas Series Format'
    assert set(df_clean['sex']) == set(scores.index), 'Values in stratum do not correspond to values in original data'
