import pandas as pd
from fastapi import FastAPI

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
Trained_model = None

@app.on_event("startup")
async def read_root():
    return {
        "message": "Data loaded from local CSV!",
        "rows_train": len(train_df),
        "rows_test": len(test_df)
    }
@app.on_event("startup")
async def train_model():
    global Trained_model
    if train_df is None:
        return {"error": "Load and clean data first!"}
    classifier.fit(train_df, target)
    Trained_model = classifier.model
    print("Model trained successfully.")
    # return {"message": "Model trained successfully."}

@app.get("/get_model/")
async def get_model():
    global Trained_model
    if Trained_model is None:
        return {"error": "Model not trained yet!"}
    print("Model retrieved successfully.")
    return Trained_model

@app.on_event("startup")
async def evaluate_model():
    if test_df is None:
        return {"error": "Load and clean data first!"}
    evaluator = Evaluator()
    results = evaluator.evaluate(classifier, test_df, target)
    print(results)
    return {"evaluation_results": results}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

