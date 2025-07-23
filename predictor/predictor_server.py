import pickle
from fastapi import FastAPI, UploadFile, Body

from predictor.predictor_model import NaiveBayesClassifier

app = FastAPI()
classifier = NaiveBayesClassifier

model = None

@app.on_event("startup")
async def load_model():
    global model
    response = requests.get("http://trainer-service:8000/get_model/")
    if response.status_code == 200:
        model = response.json()
    else:
        print("Error: Could not pull model")

@app.post("/classify_record/")
async def classify(record: dict = Body(...)):
    global model
    if not model:
        return {"error": "No model loaded yet!"}
    prediction = classifier.classify(record)
    return {"prediction": prediction.tolist()}
