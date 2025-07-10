from sklearn.model_selection import train_test_split
import pandas as pd

class DataProcessor:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        self.df.dropna(inplace=True)
        return self.df

    def split_data(self, train_size=0.7, random_state=42):
        train_df, test_df = train_test_split(self.df, train_size=train_size, random_state=random_state)
        return train_df, test_df