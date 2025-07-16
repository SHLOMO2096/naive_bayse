import pandas as pd
from io import StringIO
from fastapi import FastAPI, UploadFile, File, Body, Form
from typing_inspection.typing_objects import target

from naive_bayes_classifier import NaiveBayesClassifier
from data_procesor import DataProcessor
from evaluator import Evaluator

app = FastAPI()
classifier = NaiveBayesClassifier()
train_df = None
test_df = None
target = None

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Naive Bayes Classifier API"}

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile, target_column: str = Form(...)):
    global train_df, test_df
    global target
    contents = await file.read()

    if not contents:
        return {"error": "Uploaded file is empty!"}

    decoded = contents.decode("utf-8")
    df = pd.read_csv(StringIO(decoded))

    if target_column not in df.columns:
        return {"error": f"Column '{target_column}' not found."}


    target = target_column
    print(f"Target column is: {target}")
    processor = DataProcessor(df)
    processor.clean_data()
    train_df, test_df = processor.split_data()
    print("Data loaded, cleaned and split successfully.")
    return {"message": "Upload and processing done!"}

@app.post("/train_model/")
async def train_model():
    if train_df is None:
        return {"error": "Load and clean data first!"}
    classifier.fit(train_df, target)
    return {"message": "Model trained successfully."}

@app.get("/evaluate_model/")
async def evaluate_model():
    if test_df is None:
        return {"error": "Load and clean data first!"}
    evaluator = Evaluator()
    results = evaluator.evaluate(classifier, test_df, target)
    return {"evaluation_results": results}

@app.post("/classify_record/")
async def classify_record(record: dict = Body(...)):
    prediction = classifier.classify(record)
    return {"prediction": prediction}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

