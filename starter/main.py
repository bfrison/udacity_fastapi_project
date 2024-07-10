'''
This script defines the API which exposes the machine learning model
'''

import os
import pickle

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import pandas as pd
from pydantic import BaseModel, Field

from starter.ml.model import inference


class CensusEntry(BaseModel):
    '''
    Pydantic model mirroring a DataFrame row which is used for inference in the
    machine learning model
    '''

    age: int
    workclass: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_weel: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')


app = FastAPI()

with open(os.path.join('model', 'logistic_regression.pkl'), 'rb') as f:
    model = pickle.load(f)


@app.get('/')
async def greet() -> str:
    '''
    This get method greets the client in plain text
    '''
    return PlainTextResponse('Welcome to the salary predictor!')


@app.post('/infer/')
async def infer_salary(entries: list[CensusEntry]) -> list[str]:
    '''
    This post method accepts a DataFrame in json format (orient='records') and
    returns a list of salary predictions
    '''
    df = pd.DataFrame([entry.model_dump(by_alias=True) for entry in entries])

    y_pred = inference(model, df)

    return y_pred.to_list()
