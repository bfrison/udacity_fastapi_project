from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI()


@app.get('/')
async def greet():
    return PlainTextResponse('Welcome to the salary predictor!')
