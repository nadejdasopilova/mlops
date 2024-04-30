#!/bin/bash

echo "Создание наборов данных..."
python3 data_creation.py

echo "Предобработка данных..."
python3 model_preprocessing.py

echo "Создание и обучение модели..."
python3 model_preparation.py

echo "Тестирование модели..."
python3 model_testing.py
