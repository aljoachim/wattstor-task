import pandas as pd
from src import models
from src import utilities
import numpy as np


def test_constant_data_rnn():
    y_train = pd.Series(np.zeros(800))
    y_test = pd.Series(np.zeros(200))
    model = models.RNN(y_train, y_test)
    y_pred = model.run()
    assert (y_test == y_pred).all()

def test_actual_data_rnn():
    y_train, y_test = utilities.data_pipeline('data/SG.csv', 'Consumption', 0.8)
    model = models.RNN(y_train, y_test)
    y_pred = model.run()
    assert len(y_test) == len(y_pred)
