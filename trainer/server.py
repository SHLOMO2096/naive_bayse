import pandas as pd
from fastapi import FastAPI
import logging

from naive_bayes_classifier import NaiveBayesClassifier
from data_procesor import DataProcessor
from evaluator import Evaluator

# Set up logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler("trainer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Load dataset from CSV
df = pd.read_csv("data.csv")
classifier = NaiveBayesClassifier()
# The target column is assumed to be the last column in the CSV
target = df.columns[-1]

# Initialize data processor and clean/split data
processor = DataProcessor(df)
processor.clean_data()
train_df, test_df = processor.split_data()
Trained_model = None

# Event: On startup, print data loading info
@app.on_event("startup")
async def read_root():
    return {
        "message": "Data loaded from local CSV!",
        "rows_train": len(train_df),
        "rows_test": len(test_df)
    }

# Event: On startup, train the Naive Bayes model
@app.on_event("startup")
def train_model():
    global Trained_model
    if train_df is None:
        return {"error": "Load and clean data first!"}
    classifier.fit(train_df, target)
    Trained_model = classifier.model
    logger.info("Model trained successfully.")
    return {"message": "Model trained successfully."}

# Endpoint: Get the trained model as JSON
@app.get("/get_model/")
def get_model():
    global Trained_model
    if Trained_model is None:
        return {"error": "Model not trained yet!"}
    logger.info("Model retrieved successfully.")
    return Trained_model

# Event: On startup, evaluate the model on the test set
@app.on_event("startup")
def evaluate_model():
    if test_df is None:
        return {"error": "Load and clean data first!"}
    evaluator = Evaluator()
    results = evaluator.evaluate(classifier, test_df, target)
    logger.info(f"Evaluation results: {results}")
    return {"evaluation_results": results}

# Code for local running (currently commented out)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

