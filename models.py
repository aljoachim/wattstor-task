import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from utilities import train_test_split
import tensorflow as tf


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
        self._max_epochs = 500
        self._early_stopping_patience = 50
        self._learning_rate = 0.02
        self._sgd_momentum = 0.9
        self._val_ratio = 0.2
        self._y_all = pd.concat([y_train, y_test])

    def _scale_data(self):
        self._scale_factor = self._y_all.max()
        y_train = self._y_train / self._scale_factor
        y_test = self._y_test / self._scale_factor
        y_all = self._y_all / self._scale_factor
        return y_train, y_test, y_all
    
    def _scale_back(self, y):
        return y * self._scale_factor
    
    def _create_dataset(self, y_input, y_targets, shuffle):
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            y_input.to_numpy(),
            targets=y_targets,
            sequence_length=self._window_len,
            batch_size=self._batch_size,
            shuffle=shuffle
        )
        return dataset

    def _create_datasets(self):
        y_train, y_test, y_all = self._scale_data()
        y_train, y_val = train_test_split(y_train, 1 - self._val_ratio)
        y_train = self._create_dataset(y_train, y_train[self._window_len:], True)
        y_val = self._create_dataset(y_val, y_val[self._window_len:], False)
        y_test = self._create_dataset(
            y_all[-(self._window_len + len(y_test)):],
            y_test,
            False
        )
        return y_train, y_val, y_test
    
    def _create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, 1)),
            tf.keras.layers.SimpleRNN(self._rnn_size),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def _fit_model(self, model, y_train, y_val):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae', 
            patience=self._early_stopping_patience, 
            restore_best_weights=True
        )
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=self._learning_rate, 
            momentum=self._sgd_momentum
        )
        model.compile(
            loss=tf.keras.losses.Huber(), 
            optimizer=optimizer, 
            metrics=['mae']
        )
        model.fit(
            y_train, 
            validation_data=y_val, 
            epochs=self._max_epochs,
            callbacks=[early_stopping]
        )
        return model
    
    def _predict(self, model, y_test):
        y_pred = model.predict(y_test)
        y_pred = self._scale_back(y_pred)
        y_pred = y_pred.squeeze()
        y_pred = pd.Series(y_pred, index=self._y_test.index)
        return y_pred

    def run(self):
        if self._is_data_const(): 
            return self._y_test
        y_train, y_val, y_test = self._create_datasets()
        model = self._create_model()
        model = self._fit_model(model, y_train, y_val)
        y_pred = self._predict(model, y_test)
        return y_pred
