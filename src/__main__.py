"""Command line entry point for the call me maybe project."""

import argparse
import sys

from .parser import InputError, load_functions, load_prompts


def build_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser.

    Returns:
        The parser with the three project options.
    """
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


def main() -> int:
    """Load the inputs and report a short summary.

    Returns:
        Process exit code: 0 on success, 1 on an input error.
    """
    args = build_parser().parse_args()
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"loaded {len(functions)} function definitions")
    print(f"loaded {len(prompts)} prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
