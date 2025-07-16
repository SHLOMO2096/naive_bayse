import pandas as pd

from data_loader import DataLoader
from naive_bayes_classifier import NaiveBayesClassifier
from data_procesor import DataProcessor
from evaluator import Evaluator

class App:
    def __init__(self, target_column):
        self.classifier = NaiveBayesClassifier()
        self.train_df = None
        self.test_df = None
        self.target_column = target_column

    def load_and_clean(self):
        file_path = input("Enter data file path: ")
        df = DataLoader.load_data(file_path)
        processor = DataProcessor(df)
        processor.clean_data()
        self.train_df, self.test_df = processor.split_data()
        print("Data loaded, cleaned and split successfully.")

    def train_model(self):
        if self.train_df is None:
            print("Load and clean data first!")
            return
        self.classifier.fit(self.train_df, self.target_column)
        print("Model trained successfully.")

    def evaluate_model(self):
        if self.test_df is None:
            print("Load and clean data first!")
            return
        evaluator = Evaluator()
        evaluator.evaluate(self.classifier, self.test_df, 'p')

    def classify_record(self):
        record = {}
        for col in self.train_df:
            if col != self.target_column:
                val = input(f"Enter value for '{col}': ")
                record[col] = val
        prediction = self.classifier.classify(record)
        print(f"Prediction: {prediction}")


