import pandas as pd
from sklearn.metrics import accuracy_score
import joblib


def test_model(file_model):
    # Загрузка тестовых данных
    test_data = pd.read_csv("test/processed_test_data.csv")

    # Предполагаем, что последний столбец - это целевая переменная
    X_test = test_data[["1", "2", "4", "5"]]
    y_test = test_data[["Name"]]

    # Загрузка модели
    model = joblib.load(file_model)

    # Предсказания на тестовых данных
    predictions = model.predict(X_test)

    # Вычисление точности
    accuracy = accuracy_score(y_test, predictions)

    print(f"Точность модели {file_model} на тестовых данных: {accuracy:.2f}")


def main():
    test_model("train/model.pkl")


if __name__ == "__main__":
    main()
