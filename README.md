# Task for wattstor - time series forecast

## Description
This repository contains scripts for time series forecasting. 3 methods are implemented:
Naive Forecasting, Autoregression, and RNN. You can select one of the methods and column in data, which should be predicted. You have to also specify input file name, as well as output destination to save plot image. Data are automatically split to train and test. After each run, MAE, MSE and RMSE metrics achieved on test set are printed. If you want to see all of the training data as well, you have to specify it in the arguments.

## Install requirements
- Python 3.10 required
- Installation steps will work on Linux systems
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```


## Run prediction
```script.py [-h] --input INPUT --quantity QUANTITY --model MODEL --output OUTPUT [--plot_train PLOT_TRAIN]```

Runs time series forecast

Options:

    -h, --help            show this help message and exit
    --input INPUT, -i INPUT
                            Input file name
    --quantity QUANTITY, -q QUANTITY
                            Quantity name to predict
    --model MODEL, -m MODEL
                            Model type, one of: naive, autoreg, rnn (case insensitive)
    --output OUTPUT, -o OUTPUT
                            File to save plot to
    --plot_train PLOT_TRAIN, -pt PLOT_TRAIN
                            Plot training data (default: False)

## Run tests
- To run all tests
```
pytest
```
- To run selected test file (example)
```
pytest tests/test_autoreg.py
```
- To run selected unit test (example)
```
pytest tests/test_autoreg.py::test_constant_data_autoreg
```
