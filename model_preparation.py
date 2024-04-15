import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


def model_prep(file_data, file_model):
    # Загрузка данных
    data = pd.read_csv(file_data)

    X = data[["0", "1"]]
    y = data["day"]

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                      random_state=42)

    # Создание и обучение модели
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Сохранение модели
    joblib.dump(model, file_model)
    print("Модель обучена и сохранена как " + file_model)


def main():
    model_prep("train/processed_train_data_without_anomaly.csv",
               "train/model_without_anomaly.pkl")
    model_prep("train/processed_train_data_with_anomaly.csv",
               "train/model_with_anomaly.pkl")


if __name__ == "__main__":
    main()
