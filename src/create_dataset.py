import os
import pandas as pd
from catboost.datasets import titanic

def main():
    # Создаем папку, если она не существует
    output_dir = '../data/raw'  # Путь относительно src/
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем датасет
    train, _ = titanic()

    # Путь для сохранения файла CSV
    output_file = os.path.join(output_dir, 'titanic_dataset.csv')

    # Сохраняем датасет в файл CSV
    train.to_csv(output_file, index=False)

    print(f'Dataset сохранен в {output_file}')

if __name__ == "__main__":
    main()
