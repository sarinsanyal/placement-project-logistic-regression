# app.py
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load your saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define request body
class InputData(BaseModel):
    features: list[float]  # list of input values

# Create app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Logistic Regression API is running!"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    probs = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]
    ans = ["Won't be placed", "Will be placed"]
    return {
        "prediction": ans[prediction],
        "probabilities": probs.tolist()
    }