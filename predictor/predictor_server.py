import httpx
from fastapi import FastAPI, UploadFile, Body, requests
import predictor_model

app = FastAPI()

model = None

@app.on_event("startup")
async def load_model():
    global model
    async with httpx.AsyncClient() as client:
        response = await client.get("http://trainer:8000/get_model/")
    if response.status_code == 200:
        model = response.json()
        print(model)
    else:
        print("Error: Could not pull model")

@app.post("/classify_record/")
async def post_classify(record: dict = Body(...)):
    global model
    if not model:
        return {"error": "No model loaded yet!"}
    prediction = predictor_model.classify(model ,record)
    return {"prediction": prediction}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8001)