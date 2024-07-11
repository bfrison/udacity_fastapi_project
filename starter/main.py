'''
This script defines the API which exposes the machine learning model
'''

import os
import pickle

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from starter.ml.model import inference


class CensusEntry(BaseModel):
    '''
    Pydantic model mirroring a DataFrame row which is used for inference in the
    machine learning model
    '''

    age: int
    workclass: str | None
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str | None
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_weel: int = Field(alias='hours-per-week')
    native_country: str | None = Field(alias='native-country')


model_path = os.path.join('model', 'logistic_regression.pkl')
model = None

app = FastAPI()


def load_model() -> Pipeline | None:
    '''
    This function checks if the model is already loaded, and loads it if it is
    not the case
    '''
    global model
    if not model:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            model = None

    return model


@app.get('/')
async def greet() -> str:
    '''
    This get method greets the client in plain text
    '''
    return PlainTextResponse('Welcome to the salary predictor!')


@app.get('/check_model')
async def check_model() -> str:
    '''
    This get method checks if the model is correctly loaded
    '''
    model = load_model()
    if model:
        return PlainTextResponse('Model is properly loaded')
    else:
        return PlainTextResponse('Model is not loaded yet')


@app.post('/infer/')
async def infer_salary(entries: list[CensusEntry]) -> list[str]:
    '''
    This post method accepts a DataFrame in json format (orient='records') and
    returns a list of salary predictions
    '''
    global model
    if not model:
        model = load_model()

    df = pd.DataFrame([entry.model_dump(by_alias=True) for entry in entries])

    y_pred = inference(model, df)

    return y_pred.to_list()
