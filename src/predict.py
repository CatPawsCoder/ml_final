import pandas as pd
import joblib

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

def load_model(model_path):
    return joblib.load(model_path)

def predict(model, df):
    X = df[['Pclass', 'Sex', 'Age']]
    predictions = model.predict(X)
    return predictions

if __name__ == '__main__':
    # Загрузка данных
    df = load_data('data/raw/new_data.csv')  # Предположим, что это новые данные для предсказания
    
    # Загрузка обученного скалера
    scaler = joblib.load('models/scaler.pkl')
    
    # Предобработка данных
    df = process_data(df, scaler)
    
    # Загрузка обученной модели
    model = load_model('models/pretrained_model.pkl')
    
    # Предсказание
    predictions = predict(model, df)
    print(predictions)
