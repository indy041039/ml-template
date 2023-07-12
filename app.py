from fastapi import FastAPI
from typing import Optional
import uvicorn

import os
import glob
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

@app.get('/')
def home():
    return {'message': 'Hello World'}

@app.get('predict')
def prediction():
    return {'prediction': 0}

if __name__=='__main__':
    uvicorn.run('app:app', port=8080, reload=True)