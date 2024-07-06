'''
This script deduplicates rows from census.csv and stores the results as
clean_census.gz
'''

import os

import pandas as pd


def preprocessing(data_dir: str, input_file: str) -> pd.DataFrame:
    '''
    This function completes the preprocessing step on the input_df
    '''
    df = pd.read_csv(
        os.path.join(data_dir, input_file),
        sep=', ',
        engine='python',
        na_values='?',
    ).convert_dtypes()
    df_dedup = df.drop_duplicates()

    return df_dedup


if __name__ == '__main__':
    data_dir = 'data'
    input_file = 'census.csv'

    df_dedup = preprocessing(data_dir, input_file)

    df_dedup.to_csv(os.path.join(data_dir, 'clean_census.gz'), index=False)
