import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from src.utilities import train_test_split
import tensorflow as tf
import numpy as np

# Base model class
class ModelBase:
    def __init__(self, y_train: pd.Series, y_test: pd.Series) -> None:
        self._y_train = y_train
        self._y_test = y_test
        
    def _is_data_const(self) -> bool:
        return (self._y_train == self._y_train.iloc[0]).all()


# Class for naive forecast method
class NaiveForecast(ModelBase):
    def __init__(self, y_train: pd.Series, y_test: pd.Series) -> None:
        super().__init__(y_train, y_test)
        self._period = 48 # Period - one day
        
    # Run the training and inference
    def run(self) -> pd.Series:
        if self._is_data_const(): # Constant data causes numerical problems, so just return the test data
            return self._y_test
        y_pred = self._y_test.shift(self._period) # Shift data one day ahead
        y_pred[:self._period] = self._y_train[-self._period:] # Copy the rest from train set
        return y_pred


# Class for auto regression model
class AutoRegression(ModelBase):
    def __init__(self, y_train: pd.Series, y_test: pd.Series) -> None:
        super().__init__(y_train, y_test)
        self._max_order = 100 # Maximum number of lags
    
    # Run the training and inference
    def run(self) -> pd.Series:
        if self._is_data_const(): # Constant data causes numerical problems, so just return the test data
            return self._y_test
        y_test_index = self._y_test.index
        y_pred = []
        order = len(ar_select_order(self._y_train.values, self._max_order).ar_lags) # Find optimal order (number of lags)
        model = AutoReg(self._y_train.values, lags=order) # Create and fit the model
        model = model.fit()
        for y_true_single in self._y_test:
            y_pred_single = model.forecast()[0] # Get forecast and append to result
            y_pred.append(y_pred_single)
            model = model.append([y_true_single], refit=False) # Add actual observed value to model
        y_pred = pd.Series(y_pred, index=y_test_index) # Add time index
        return y_pred


# Class for RNN model
class RNN(ModelBase):
    def __init__(self, y_train: pd.Series, y_test: pd.Series) -> None:
        super().__init__(y_train, y_test)
        # Various hyperparameters
        self._batch_size = 32
        self._window_len = 1 # Length of history window that model is allowed to look at
        self._rnn_size = 96
        self._max_epochs = 500
        self._early_stopping_patience = 50
        self._learning_rate = 0.02
        self._sgd_momentum = 0.9
        self._val_ratio = 0.2
        self._y_all = pd.concat([y_train, y_test])

    # Scales data to 0;1 range
    def _scale_data(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        self._scale_factor = self._y_all.max()
        y_train = self._y_train / self._scale_factor
        y_test = self._y_test / self._scale_factor
        y_all = self._y_all / self._scale_factor
        return y_train, y_test, y_all
    
    # Scale data back to original range
    def _scale_back(self, y: np.ndarray) -> np.ndarray:
        return y * self._scale_factor
    
    # Create tensorflow dataset from input series, as a list of samples from past and the following sample
    def _create_dataset(self, y_input: pd.Series, y_targets: pd.Series, shuffle: bool) -> tf.data.Dataset:
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            y_input.to_numpy(),
            targets=y_targets,
            sequence_length=self._window_len,
            batch_size=self._batch_size,
            shuffle=shuffle
        )
        return dataset

    # Create all datasets from inputs
    def _create_datasets(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
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
    
    # Define RNN model
    def _create_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, 1)),
            tf.keras.layers.SimpleRNN(self._rnn_size),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    # Fit and return RNN model
    def _fit_model(self, model: tf.keras.Model, y_train: tf.data.Dataset, y_val: tf.data.Dataset) -> tf.keras.Model:
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
    
    # Predict the test data
    def _predict(self, model: tf.keras.Model, y_test: tf.data.Dataset) -> pd.Series:
        y_pred = model.predict(y_test)
        y_pred = self._scale_back(y_pred)
        y_pred = y_pred.squeeze()
        y_pred = pd.Series(y_pred, index=self._y_test.index)
        return y_pred

    # Run the training and inference
    def run(self) -> pd.Series:
        if self._is_data_const(): # Constant data causes numerical problems, so just return the test data
            return self._y_test
        y_train, y_val, y_test = self._create_datasets() # Create tensorfloe datasets as shifting windows
        model = self._create_model() # Get TF model
        model = self._fit_model(model, y_train, y_val) # Fit the model
        y_pred = self._predict(model, y_test) # Make predictions on test set
        return y_pred
