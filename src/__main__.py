import argparse

from src import parser
from src.constrained_decoding import run


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parser.parse(
        functions_definition=args.functions_definition,
        input=args.input,
        output=args.output,
    )
    run(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        exit()
