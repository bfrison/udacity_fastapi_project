import pytest
import yaml

from starter.ml.data import process_data
from starter.ml.model import create_pipeline, train_model

scores = ['f1_score', 'precision_score', 'recall_score']


@pytest.fixture
def data_dir():
    return 'data'


@pytest.fixture
def input_file():
    return 'census.csv'


@pytest.fixture
def df_clean(data_dir, input_file):
    return process_data(data_dir, input_file).head(10)


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
