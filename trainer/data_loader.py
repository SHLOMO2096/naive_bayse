import pandas as pd

class DataLoader:
    @staticmethod
    def load_data(file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def load_json(file_path):
        return pd.read_json(file_path)

    @staticmethod
    def load_excel(file_path):
        return pd.read_excel(file_path)

    @staticmethod
    def load_from_df(df: pd.DataFrame):
        return df

