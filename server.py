import pandas as pd
from fastapi import FastAPI, Body

from naive_bayes_classifier import NaiveBayesClassifier
from data_procesor import DataProcessor
from evaluator import Evaluator

app = FastAPI()
df = pd.read_csv("data.csv")
classifier = NaiveBayesClassifier()
target = df.columns[-1]

processor = DataProcessor(df)
processor.clean_data()
train_df, test_df = processor.split_data()


@app.on_event("startup")
async def read_root():
    return {
        "message": "Data loaded from local CSV!",
        "rows_train": len(train_df),
        "rows_test": len(test_df)
    }
@app.on_event("startup")
async def train_model():
    if train_df is None:
        return {"error": "Load and clean data first!"}
    classifier.fit(train_df, target)
    return {"message": "Model trained successfully."}

@app.on_event("startup")
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



