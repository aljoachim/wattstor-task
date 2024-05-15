from src.utilities import data_pipeline, calculate_metrics, plot_results, print_metrics
from src.models import NaiveForecast, AutoRegression, RNN
from src.parse_args import parse_args

# Mapping models to argument params
model_map = {
    'naive': NaiveForecast,
    'autoreg': AutoRegression,
    'rnn': RNN
}

# Default train to test ratio
train_ratio = 0.8

def main() -> None:
    args = parse_args(list(model_map.keys())) # Parse input arguments
    y_train, y_test = data_pipeline(args.input, args.quantity, train_ratio) # Preprocess data and split to train/test
    model_cls = model_map[args.model] # Select model based on input
    model = model_cls(y_train, y_test) # Initialize model
    y_pred = model.run() # Fit model and make predictions for test data
    metrics = calculate_metrics(y_test, y_pred) # Calculate metrics for prediction
    print_metrics(metrics) # Print resulting metrics
    plot_results(y_train, y_test, y_pred, args.quantity, args.output, args.plot_train) # Generate and save the plot

if __name__ == '__main__':
    main()
