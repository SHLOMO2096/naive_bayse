# FastAPI server for classifying records using a Naive Bayes model
import httpx
from fastapi import FastAPI, UploadFile, Body, requests
import predictor_model

app = FastAPI()

model = None  # The model will be loaded from the external server

@app.on_event("startup")
async def load_model():
    global model
    # Fetch the model from the external (trainer) server
    async with httpx.AsyncClient() as client:
        response = await client.get("http://trainer:8000/get_model/")
    if response.status_code == 200:
        model = response.json()
        print(model)
    else:
        print("Error: Could not pull model")

@app.post("/classify_record/")
def post_classify(record: dict = Body(...)):
    global model
    # Check if the model is loaded
    if not model:
        return {"error": "No model loaded yet!"}
    # Classify the record using the model
    prediction = predictor_model.classify(model ,record)
    return {"prediction": prediction}

# Code for local running (currently commented out)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8001)