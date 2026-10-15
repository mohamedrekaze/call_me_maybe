import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

FUNCTIONS: list[Any] = []
PROMPTS: list[Any] = []
FUNCTIONS_DICT: Dict[str, str] = {}


class ValueSpec(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ValueSpec] = Field(default_factory=dict)
    returns: ValueSpec


class Prompt(BaseModel):
    prompt: str = Field(min_length=1)


def parse(functions_definition: str, input: str, output: str) -> None:
    try:
        validate_function_definition(functions_definition)
        validate_input(input)
        validate_output(output)
    except Exception as e:
        raise e


def validate_function_definition(path: str) -> None:
    file_path = Path(path)
    with file_path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{file_path}: expected a JSON array")
    for entry in data:
        FunctionDefinition.model_validate(entry)
        FUNCTIONS.append(entry)
        FUNCTIONS_DICT[entry["name"]] = entry


def validate_input(path: str) -> None:
    file_path = Path(path)
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{file_path}: expected a JSON array")
    for entry in data:
        Prompt.model_validate(entry)
        PROMPTS.append(entry)


def validate_output(path: str) -> None:
    file_path = Path(path)
    if file_path.is_file() and file_path.stat().st_size > 0:
        raise ValueError(f"{file_path}: has existing content")
