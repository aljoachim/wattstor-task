import argparse


class OptionalDefaultFormatted(argparse.HelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        help_str = action.help
        if not action.required:
            help_str = help_str + f' (default: {action.default})'
        return help_str

    
def validate_model(model: str, allowed_models: list[str]) -> None:
    if model not in allowed_models:
        raise ValueError(f'Model type must be one of: {", ".join(allowed_models)} (case insensitive)')

def parse_args(models: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Runs time series forecast', formatter_class=OptionalDefaultFormatted)
    
    parser.add_argument('--input', '-i', type=str, required=True, help='Input file name')
    parser.add_argument('--quantity', '-q', type=str, required=True, help='Quantity name to predict')
    parser.add_argument('--model', '-m', type=lambda x: str(x).lower(), required=True, help=f'Model type, one of: {", ".join(models)} (case insensitive)')
    parser.add_argument('--output', '-o', type=str, required=True, help='File to save plot to')
    parser.add_argument('--plot_train', '-pt', type=bool, default=False, help='Plot training data')

    args = parser.parse_args()
    validate_model(args.model, models)
    return args
