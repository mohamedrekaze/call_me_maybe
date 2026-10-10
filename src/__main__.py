import argparse
from pprint import pprint
# from llm_sdk import Small_LLM_Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small LLM Model")

    parser.add_argument(
        "--functions_definitions",
        help="path to the function definitions file",
        default="data/input/functions_definition.json",
    )

    parser.add_argument(
        "--input",
        help="path to the input file",
        default="data/input/function_calling_tests.json",
    )

    parser.add_argument(
        "--output",
        help="path to the output file",
        default="data/output/function_calls.json",
    )

    args = parser.parse_args()
    pprint(vars(args))
