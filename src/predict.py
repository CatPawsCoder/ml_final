import pandas as pd
import joblib
import os
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    return pd.read_csv(file_path)

def load_model(model_path):
    return joblib.load(model_path)

def predict(model, df):
    X = df[['Pclass', 'Sex', 'Age']]
    predictions = model.predict(X)
    return predictions

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

def save_predictions(predictions, file_path):
    pd.DataFrame(predictions, columns=['Prediction']).to_csv(file_path, index=False)
    print(f'Predictions saved to {file_path}')

def main():
    # Load processed data for prediction
    file_path = '../data/processed/processed_data.csv'
    df = load_data(file_path)
    
    # Load trained model
    model = load_model('../models/pretrained_model.pkl')
    
    # Load true labels (Survived column)
    y_true = df['Survived']
    
    # Make predictions
    predictions = predict(model, df)
    
    # Save predictions to file
    predictions_file = '../predictions/predictions.csv'
    save_predictions(predictions, predictions_file)

    # Calculate MSE
    mse = calculate_metrics(y_true, predictions)
    print(f'Mean Squared Error (MSE): {mse:.4f}')

if __name__ == '__main__':
    main()

