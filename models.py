import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


class ModelBase:
    def __init__(self, y_train: pd.Series, y_test: pd.Series) -> None:
        self._y_train = y_train
        self._y_test = y_test
        
    def _is_data_const(self) -> bool:
        return (self._y_train == self._y_train.iloc[0]).all()


class NaiveForecast(ModelBase):
    def __init__(self, y_train: pd.Series, y_test: pd.Series) -> None:
        super().__init__(y_train, y_test)
        self._period = 48
        
    def run(self) -> pd.Series:
        if self._is_data_const(): 
            return self._y_test
        y_pred = self._y_test.shift(self._period)
        y_pred[:self._period] = self._y_train[-self._period:]
        return y_pred


class AutoRegression(ModelBase):
    def __init__(self, y_train: pd.Series, y_test: pd.Series) -> None:
        super().__init__(y_train, y_test)
        self._max_order = 100
    
    def run(self) -> pd.Series:
        if self._is_data_const(): 
            return self._y_test
        y_test_index = self._y_test.index
        y_pred = []
        order = len(ar_select_order(self._y_train.values, self._max_order).ar_lags)
        model = AutoReg(self._y_train.values, lags=order)
        model = model.fit()
        for y_true_single in self._y_test:
            y_pred_single = model.forecast()[0]
            y_pred.append(y_pred_single)
            model = model.append([y_true_single], refit=False)
        y_pred = pd.Series(y_pred, index=y_test_index)
        return y_pred


class RNN(ModelBase):
    def __init__(self, y_train: pd.Series, y_test: pd.Series) -> None:
        super().__init__(y_train, y_test)
        self._batch_size = 32
        self._window_len = 1
        self._rnn_size = 96
        
    def run(self):
        pass
