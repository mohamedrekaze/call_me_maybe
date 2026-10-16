import argparse
import sys

from src import parser
from src.constrained_decoding import run
from src.runner import DEFAULT_MODEL, SECONDARY_MODEL


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser with the project options."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate prompts into function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        metavar="<function_definition_file>",
        default="data/input/functions_definition.json",
        help="path to the function definitions file",
    )
    parser.add_argument(
        "--input",
        metavar="<input_file>",
        default="data/input/function_calling_tests.json",
        help="path to the prompts file",
    )
    parser.add_argument(
        "--output",
        metavar="<output_file>",
        default="data/output/function_calls.json",
        help="path to the output file",
    )
    parser.add_argument(
        "--model",
        metavar="<model_name>",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face model to run "
            f"(default: {DEFAULT_MODEL}, "
            f"also supported: {SECONDARY_MODEL})"
        ),
    )
    return parser


def main() -> None:
    """Validate the inputs and decode every prompt."""
    args = build_parser().parse_args()
    parser.parse(
        functions_definition=args.functions_definition,
        input=args.input,
        output=args.output,
    )
    run(args.output, model_name=args.model)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
