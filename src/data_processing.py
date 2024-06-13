import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def process_data(df):
    # Добавьте код для обработки данных
    return df

if __name__ == '__main__':
    df = load_data('data/raw/titanic.csv')
    df = process_data(df)
    print(df.head())
