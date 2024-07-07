import pandas as pd
import pytest
from preprocessing import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import yaml

from starter.ml.model import compute_model_metrics, create_pipeline, inference, score_strata, train_model

scores = ['f1_score', 'precision_score', 'recall_score']

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
