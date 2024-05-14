import pandas as pd
from pandas.api.types import is_numeric_dtype
from  sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy import sqrt
import matplotlib.pyplot as plt


def train_test_split(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    length = len(df)
    train_len = int(length * train_ratio)
    return df[:train_len], df[train_len:]

def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if is_numeric_dtype(df[column].dtype):
            df[column] = df[column].interpolate(method='slinear')
    return df

def load_csv(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=';')
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df = impute_data(df)
    df = df.set_index('Time')
    return df

def validate_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f'Quantity {column} does not exist in dataset, available quantities are: {", ".join(df.columns)}')

def data_pipeline(filename: str, target: str, train_ratio: float) -> tuple[pd.Series, pd.Series]:
    df = load_csv(filename)
    df = preprocess_data(df)
    validate_column(df, target)
    df_train, df_test = train_test_split(df, train_ratio)
    y_train, y_test = df_train[target], df_test[target]
    return y_train, y_test

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

def plot_results(y_train: pd.Series, y_test: pd.Series, y_pred: pd.Series, target: str, plot_train: bool) -> None:
    plt.figure(figsize=(10, 6))
    if plot_train:
        y_all = pd.concat([y_train, y_test])
        plt.plot(y_all.index, y_all, label='True')
    else:
        plt.plot(y_test.index, y_test, label='True')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel(target)
    plt.title(f'{target} prediction')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

def print_metrics(metrics: dict[str, float]) -> None:
    print(f'MAE: {metrics["mae"]:.2f}')
    print(f'MSE: {metrics["mse"]:.2f}')
    print(f'RMSE: {metrics["rmse"]:.2f}')
