import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def process_data(df, scaler):
    # Заполнение пропусков
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Преобразование категориальных переменных
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Нормализация данных
    df[['Pclass', 'Age']] = scaler.transform(df[['Pclass', 'Age']])
    
    return df

if __name__ == '__main__':
    # Загрузка и обработка данных
    df = load_data('data/raw/titanic_dataset.csv')
    scaler = joblib.load('models/scaler.pkl')
    df = process_data(df, scaler)
    print(df.head())
