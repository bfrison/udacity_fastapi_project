import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from conftest import scores
from starter.ml.model import compute_model_metrics, inference, score_strata


def test_preprocessing_dataframe(df_clean):

    assert isinstance(
        df_clean, pd.DataFrame
    ), 'Data is not in the Pandas DataFrame format'
    assert len(df_clean) > 0, 'Data contains now rows'
    assert (
        'salary' in df_clean
    ), 'Data does not contain the \'salary\' target column'


def test_preprocessing_deduplication(df_clean):

    num_duplicates = df_clean.duplicated().sum()

    assert (
        num_duplicates == 0
    ), f'There are {num_duplicates:,d} duplicates in data'


def test_create_pipeline(pipeline):

    assert isinstance(
        pipeline, Pipeline
    ), 'Model is not an instance of Sci-Kit Learn Pipeline'
    assert isinstance(
        pipeline[-1], LogisticRegression
    ), 'Model\'s regressor is not an instance of LogisticRegressor'


def test_train(trained_pipeline):

    assert (
        trained_pipeline.__sklearn_is_fitted__()
    ), 'Model is not fitted after training'


@pytest.mark.parametrize('score', scores)
def test_compute_model_metrics(trained_pipeline, df_clean, salary, score):

    score_vals = compute_model_metrics(trained_pipeline, df_clean, salary)

    assert f'{score}' in score_vals, f'{score} not in score values'
    assert isinstance(score_vals[f'{score}'], float), f'{score} is not a float'


def test_inference(trained_pipeline, df_clean, salary):

    y_pred = inference(trained_pipeline, df_clean)

    assert isinstance(
        y_pred, pd.Series
    ), 'Inferred data is not in the Pandas Series format'
    assert set(y_pred) == set(
        salary
    ), 'The set of inferred values does not correspond to original target data'


@pytest.mark.parametrize('column', ['sex', 'race', 'workclass'])
def test_score_strata_vals(trained_pipeline, df_clean, salary, column):

    columns_dict = score_strata(trained_pipeline, df_clean, salary, [column])
    stratum_dict = columns_dict[column]

    assert isinstance(
        stratum_dict, dict
    ), 'Scores are not in the dictionary format'
    assert set(df_clean[column]) == set(stratum_dict.keys()), (
        'Values in stratum do not correspond to values in original'
        f'`{column}` data'
    )


@pytest.mark.parametrize('score', scores)
def test_score_strata_scores(trained_pipeline, df_clean, salary, score):

    columns_dict = score_strata(trained_pipeline, df_clean, salary, ['sex'])
    stratum_dict = columns_dict['sex']
    scores_dict = next(iter(stratum_dict.values()))

    assert f'{score}' in scores_dict, f'{score} not in score values'
    assert isinstance(scores_dict[f'{score}'], float), f'{score} is not a float'
