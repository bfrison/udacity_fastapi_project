'''
The script defines the feature engineering steps to be applied on the cleaned
dataframe
'''

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
import yaml

with open('parameters.yaml', encoding='utf-8') as f:
    params = yaml.safe_load(f)

cat_cols = params['columns']['categorical']
num_cols = params['columns']['numerical']


def apply_feature_engineering(df):
    '''
    This function applies a one hot encoder on categorical columns, and a
    standard scaler on numerical columns
    '''
    col_transf = ColumnTransformer(
        [
            ('one_hot_encoder', OneHotEncoder(sparse_output=False), cat_cols),
            ('standard_scaler', StandardScaler(), num_cols),
            (
                'target',
                FunctionTransformer(
                    lambda vals: (vals == '>50K').astype(int),
                    feature_names_out='one-to-one',
                ),
                ['salary'],
            ),
        ]
    )
    col_transf.fit(df)
    output_cols = col_transf.get_feature_names_out()
    df_eng = pd.DataFrame(
        data=col_transf.transform(df), columns=output_cols, index=df.index
    ).convert_dtypes()

    return df_eng


if __name__ == '__main__':
    df = pd.read_csv('data/clean_census.gz')
    df_eng = apply_feature_engineering(df)
    df_eng.to_csv('data/eng_census.gz', index=False)
