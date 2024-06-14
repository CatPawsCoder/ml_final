import os
import pytest
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from predict import load_data, load_model, predict, calculate_metrics, save_predictions
from sklearn.metrics import mean_squared_error

# Фикстура для подготовки данных и модели
@pytest.fixture
def setup_data_and_model():
    file_path = '../data/processed/processed_data.csv'
    df = load_data(file_path)
    model_path = '../models/pretrained_model.pkl'
    model = load_model(model_path)
    return df, model

# Тесты

def test_load_data():
    file_path = '../data/processed/processed_data.csv'
    df = load_data(file_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

def test_load_model():
    model_path = '../models/pretrained_model.pkl'
    model = load_model(model_path)
    assert model is not None  # Проверяем, что модель была успешно загружена

def test_predict(setup_data_and_model):
    df, model = setup_data_and_model
    predictions = predict(model, df)
    assert len(predictions) == len(df)
    # Можно добавить дополнительные проверки на формат предсказаний или их значения

def test_calculate_metrics(setup_data_and_model):
    df, model = setup_data_and_model
    y_true = df['Survived']
    predictions = predict(model, df)
    mse = calculate_metrics(y_true, predictions)
    assert mse >= 0  # Проверяем, что значение MSE неотрицательно

def test_save_predictions(setup_data_and_model, tmpdir):
    df, model = setup_data_and_model
    predictions = predict(model, df)
    predictions_file = os.path.join(tmpdir, 'predictions.csv')
    save_predictions(predictions, predictions_file)
    assert os.path.isfile(predictions_file)  # Проверяем, что файл с предсказаниями был сохранён

if __name__ == "__main__":
    pytest.main()
