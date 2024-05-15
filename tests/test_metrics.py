import pandas as pd
from src import utilities
import numpy as np


def test_metrics():
    a = pd.Series(np.zeros(5))
    b = pd.Series(np.arange(5))
    metrics = utilities.calculate_metrics(a, b)
    assert np.isclose(metrics['mae'], 2)
    assert np.isclose(metrics['mse'], 6)
    assert np.isclose(metrics['rmse'], np.sqrt(6))
