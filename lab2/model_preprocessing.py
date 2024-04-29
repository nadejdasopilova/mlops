import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Загрузка данных из CSV файла."""
    return pd.read_csv(filepath)


def preprocess_data(data):
    """Предобработка данных с помощью StandardScaler."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def save_data(data, filepath):
    """Сохранение обработанных данных в новый CSV файл."""
    pd.DataFrame(data).to_csv(filepath, index=False)


def process_data(dataset_path, processed_dataset_path):
    # Загрузка данных
    data = load_data(dataset_path)

    X = data[["Miles_per_Gallon", "Cylinders", "Displacement", "Horsepower", "Weight_in_lbs", "Acceleration"]]
    y = data[["Name", "Year", "Origin"]]

    # Предобработка данных
    X_scaled = preprocess_data(X)

    # Объединяем обратно масштабированные признаки с метками
    processed_data = pd.concat(
        [y.reset_index(drop=True), pd.DataFrame(X_scaled)], axis=1)

    # Сохранение обработанных данных
    save_data(processed_data, processed_dataset_path)

    print("Предобработка данных завершена. Обработанные данные сохранены в:",
          processed_dataset_path)


def main():
    process_data("train/train_data.csv",
                 "train/processed_train_data.csv")
    process_data("test/test_data.csv",
                 "test/processed_test_data.csv")


if __name__ == "__main__":
    main()
