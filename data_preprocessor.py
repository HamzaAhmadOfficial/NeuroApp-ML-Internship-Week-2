# Task 2.4 — OOP for ML

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, path):
        self.path = path

    def load_data(self):
        self.df = pd.read_csv(self.path)
        return self.df

    def handle_missing_values(self):
        self.df.fillna(self.df.median(numeric_only=True), inplace=True)

    def encode_categorical(self):
        le = LabelEncoder()
        for col in self.df.select_dtypes(include='object'):
            self.df[col] = le.fit_transform(self.df[col])

    def scale_features(self):
        scaler = StandardScaler()
        num_cols = self.df.select_dtypes(include='number').columns
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])

    def split_data(self):
        X = self.df.drop('Survived', axis=1)
        y = self.df['Survived']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def save_processed_data(self):
        self.df.to_csv("processed_titanic.csv", index=False)

if __name__ == "__main__":
    dp = DataPreprocessor("titanic_cleaned.csv")
    dp.load_data()
    dp.handle_missing_values()
    dp.encode_categorical()
    dp.scale_features()
    dp.save_processed_data()
