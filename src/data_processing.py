import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def process_data(df, scaler):
    # Заполнение пропусков в 'Age' медианой
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Преобразование категориальных переменных 'Sex' и 'Embarked'
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(df['Embarked'].mode()[0]).astype(int)
    
    # Нормализация числовых признаков 'Pclass' и 'Age'
    df[['Pclass', 'Age']] = scaler.transform(df[['Pclass', 'Age']])
    
    return df

if __name__ == '__main__':
    # Загрузка данных
    df = load_data('data/raw/titanic_dataset.csv')
    
    # Создание и загрузка скалера
    scaler = StandardScaler()
    scaler.fit(df[['Pclass', 'Age']])
    
    # Предобработка данных
    df = process_data(df, scaler)
    
    print(df.head())
