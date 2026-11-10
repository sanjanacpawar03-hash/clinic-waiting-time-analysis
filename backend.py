from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

model = None

def train_model():
    global model
    data = pd.read_csv("data.csv")

    X = data[['hour','doc_code']]
    y = data['waiting_time']

    model = RandomForestRegressor().fit(X,y)

@app.get("/")
def home():
    return {"status":"running"}

@app.get("/train")
def train():
    train_model()
    return {"message":"Model trained"}

@app.get("/predict")
def predict(hour:int, doc:int):
    if model is None:
        return {"error":"Model not trained"}

    pred = model.predict([[hour,doc]])[0]
    return {"waiting_time": float(pred)}