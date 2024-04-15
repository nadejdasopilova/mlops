import numpy as np
import pandas as pd
import os


def generate_temperature_data(days=365, anomaly=False):
    """Генерация данных о температуре за указанное количество дней.
    Может включать аномалии."""
    base_temperature_day = 10  # Базовая температура днем
    base_temperature_night = 2  # Базовая температура ночью
    daily_variation = np.random.normal(0, 2, size=days)  # Ежедневные колебания

    if anomaly:
        anomaly_days = np.random.choice(range(days), size=int(days * 0.05),
                                        replace=False)
        daily_variation[anomaly_days] += np.random.normal(20, 5,
                                                          size=len(
                                                              anomaly_days))

    temperature_day = base_temperature_day + daily_variation + np.sin(
        np.linspace(0, 2 * np.pi, days)
    ) * 10

    temperature_night = base_temperature_night + daily_variation + np.sin(
        np.linspace(0, 2 * np.pi, days)
    ) * 10

    return temperature_day, temperature_night


def save_data(folder, data, name):
    """Сохранение данных в CSV файл."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, f"{name}.csv")
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def main():
    np.random.seed(42)  # Для воспроизводимости результатов

    # Генерация тренировочных и тестовых данных
    train_data_without_anomaly_day, train_data_without_anomaly_night = (
        generate_temperature_data(days=365))
    train_data_with_anomaly_day, train_data_with_anomaly_night = (
        generate_temperature_data(days=365, anomaly=True))
    test_data_day, test_data_night = generate_temperature_data(days=90)

    # Сохранение данных
    save_data("train",
              pd.DataFrame({
                  "day": np.arange(1, len(train_data_without_anomaly_day) + 1),
                  "temperature_day": train_data_without_anomaly_day,
                  "temperature_night": train_data_without_anomaly_night,
              }),
              "train_data_without_anomaly")
    save_data("train",
              pd.DataFrame({
                  "day": np.arange(1, len(train_data_with_anomaly_day) + 1),
                  "temperature_day": train_data_with_anomaly_day,
                  "temperature_night": train_data_with_anomaly_night,
              }),
              "train_data_with_anomaly")
    save_data("test",
              pd.DataFrame({
                  "day": np.arange(1, len(test_data_day) + 1),
                  "temperature_day": test_data_day,
                  "temperature_night": test_data_night,
              }),
              "test_data")


if __name__ == "__main__":
    main()
