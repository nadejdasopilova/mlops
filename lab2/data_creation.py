import os
from vega_datasets import data
from altair import load_dataset


def load_data():
    dataset = load_dataset("cars")
    return dataset


def save_data(folder, data, name):
    """Сохранение данных в CSV файл."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, f"{name}.csv")
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def main():
    print(data.list_datasets())
    dataset = load_data()
    test_dataset = dataset.iloc[-int(len(dataset) * 0.1):]

    # Сохранение данных
    save_data("train",
              dataset,
              "train_data")

    save_data("test",
              test_dataset,
              "test_data")


if __name__ == "__main__":
    main()
