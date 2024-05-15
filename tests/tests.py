import pytest
from src import utilities
from src import models
import numpy as np
import pandas as pd


def test_non_existent_file():
    with pytest.raises(FileNotFoundError):
        _, _ = utilities.data_pipeline('NonExistentFile.csv', 'target', 0.8)

def test_non_existent_column():
    with pytest.raises(ValueError):
        _, _ = utilities.data_pipeline('../data/SG.csv', 'random_column', 0.8)

def test_constant_data_naive():
    y_train = pd.Series(np.zeros(800))
    y_test = pd.Series(np.zeros(200))
    model = models.NaiveForecast(y_train, y_test)
    y_pred = model.run()
    assert (y_test == y_pred).all()

def test_constant_data_autoreg():
    y_train = pd.Series(np.zeros(800))
    y_test = pd.Series(np.zeros(200))
    model = models.AutoRegression(y_train, y_test)
    y_pred = model.run()
    assert (y_test == y_pred).all()

def test_constant_data_rnn():
    y_train = pd.Series(np.zeros(800))
    y_test = pd.Series(np.zeros(200))
    model = models.RNN(y_train, y_test)
    y_pred = model.run()
    assert (y_test == y_pred).all()

def test_actual_data_naive():
    y_train, y_test = utilities.data_pipeline('../data/SG.csv', 'Consumption', 0.8)
    model = models.NaiveForecast(y_train, y_test)
    y_pred = model.run()
    assert len(y_test) == len(y_pred)

def test_actual_data_autoreg():
    y_train, y_test = utilities.data_pipeline('../data/SG.csv', 'Consumption', 0.8)
    model = models.AutoRegression(y_train, y_test)
    y_pred = model.run()
    assert len(y_test) == len(y_pred)

def test_actual_data_rnn():
    y_train, y_test = utilities.data_pipeline('../data/SG.csv', 'Consumption', 0.8)
    model = models.RNN(y_train, y_test)
    y_pred = model.run()
    assert len(y_test) == len(y_pred)

def test_metrics():
    a = pd.Series(np.zeros(5))
    b = pd.Series(np.arange(5))
    metrics = utilities.calculate_metrics(a, b)
    assert np.isclose(metrics['mae'], 2)
    assert np.isclose(metrics['mse'], 6)
    assert np.isclose(metrics['rmse'], np.sqrt(6))
