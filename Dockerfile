# Используем базовый образ Python
FROM python:3.8-slim

# Устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src/ /app/src/
COPY data/ /app/data/

WORKDIR /app

# Запускаем скрипт при старте контейнера
CMD ["python", "src/data_processing.py"]
