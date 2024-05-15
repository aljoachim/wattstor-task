from utilities import data_pipeline, calculate_metrics, plot_results, print_metrics
from models import NaiveForecast, AutoRegression, RNN
from parse_args import parse_args

model_map = {
    'naive': NaiveForecast,
    'autoreg': AutoRegression,
    'rnn': RNN
}

train_ratio = 0.8

def main() -> None:
    args = parse_args(list(model_map.keys()))
    y_train, y_test = data_pipeline(args.input, args.quantity, train_ratio)
    model_cls = model_map[args.model]
    model = model_cls(y_train, y_test)
    y_pred = model.run()
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics)
    plot_results(y_train, y_test, y_pred, args.quantity, args.plot_train)

if __name__ == '__main__':
    main()
