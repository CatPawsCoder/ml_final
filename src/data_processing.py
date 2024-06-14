import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def process_data(df, scaler):
    # Handling missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Encoding categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Normalize numerical features
    df[['Pclass', 'Age']] = scaler.transform(df[['Pclass', 'Age']])

    # Encoding 'Embarked'
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    return df

def main():
    # Load data
    file_path = '../data/raw/titanic_dataset.csv'
    df = load_data(file_path)

    # Initialize and fit scaler
    scaler = StandardScaler()
    scaler.fit(df[['Pclass', 'Age']])

    # Save scaler
    os.makedirs('../models', exist_ok=True)
    joblib.dump(scaler, '../models/scaler.pkl')

    # Process data
    df = process_data(df, scaler)

    # Save processed data
    processed_data_path = '../data/processed/processed_data.csv'
    df.to_csv(processed_data_path, index=False)
    print(f'Processed data saved to {processed_data_path}')

if __name__ == '__main__':
    main()

