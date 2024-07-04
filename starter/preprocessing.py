'''
This script deduplicates rows from census.csv and stores the results as clean_census.csv
'''

import os

import pandas as pd


if __name__ == '__main__':
    data_dir = 'data'
    df = pd.read_csv(
        os.path.join(data_dir, 'census.csv'),
        sep=', ',
        engine='python',
        na_values='?',
    ).convert_dtypes()
    df_dedup = df.drop_duplicates()
    df_dedup.to_csv(os.path.join(data_dir, 'clean_census.gz'), index=False)
