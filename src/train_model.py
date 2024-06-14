import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

def load_processed_data(file_path):
    return pd.read_csv(file_path)

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Создание и обучение модели
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Оценка модели
    accuracy = model.score(X_test, y_test)
    print(f'Model accuracy: {accuracy:.2f}')
    
    return model

if __name__ == '__main__':
    # Загрузка предобработанных данных
    processed_data_file = 'data/processed/processed_data.csv'
    df = load_processed_data(processed_data_file)
    
    # Разделение данных на признаки и целевую переменную
    X = df[['Pclass', 'Sex', 'Age']]
    y = df['Survived']  # Предположим, что у вас есть колонка 'Survived'
    
    # Обучение модели
    model = train_model(X, y)
    
    # Сохранение обученной модели
    os.makedirs('models', exist_ok=True)
    model_filename = 'models/pretrained_model.pkl'
    joblib.dump(model, model_filename)

