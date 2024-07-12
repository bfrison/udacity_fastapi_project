'''
This module queries the live server with a post request containing two rows of
the input dataframe, and prints the response's details
'''
import argparse
import json

import pandas as pd
import requests


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('domain')
    args = parser.parse_args()

    df_clean = pd.read_csv('data/clean_census.gz').convert_dtypes()
    data = df_clean.head(2).to_dict(orient='records')

    response = requests.post(f'http://{args.domain}/infer', json=data)

    print(f'The request\'s body is {json.dumps(data)}')

    print(f'The response\'s status code is {response.status_code:d}')

    print(f'The response\'s content is {response.text}')
