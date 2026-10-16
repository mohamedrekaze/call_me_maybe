import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

FUNCTIONS: list[Any] = []
PROMPTS: list[Any] = []
FUNCTIONS_DICT: Dict[str, str] = {}


class ValueSpec(BaseModel):
    """Schema of one parameter or return value."""

    type: str


class FunctionDefinition(BaseModel):
    """One function available to the model."""

    name: str
    description: str
    parameters: dict[str, ValueSpec] = Field(default_factory=dict)
    returns: ValueSpec


class Prompt(BaseModel):
    """One natural-language request to process."""

    prompt: str = Field(min_length=1)


def parse(functions_definition: str, input: str, output: str) -> None:
    """Load and validate both input files and the output path."""
    validate_function_definition(functions_definition)
    validate_input(input)
    validate_output(output)


def _read_json(file_path: Path) -> Any:
    """Read a JSON file, reporting unreadable or malformed files clearly."""
    try:
        with file_path.open(encoding="utf-8") as file:
            return json.load(file)
    except OSError as exc:
        raise ValueError(f"{file_path}: cannot read file ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file_path}: invalid JSON ({exc})") from exc


def validate_function_definition(path: str) -> None:
    """Validate *path* and append every function definition to FUNCTIONS."""
    file_path = Path(path)
    data = _read_json(file_path)
    if not isinstance(data, list):
        raise ValueError(f"{file_path}: expected a JSON array")
    for entry in data:
        FunctionDefinition.model_validate(entry)
        FUNCTIONS.append(entry)
        FUNCTIONS_DICT[entry["name"]] = entry


def validate_input(path: str) -> None:
    """Validate *path* and append every prompt to PROMPTS."""
    file_path = Path(path)
    data = _read_json(file_path)
    if not isinstance(data, list):
        raise ValueError(f"{file_path}: expected a JSON array")
    for entry in data:
        Prompt.model_validate(entry)
        PROMPTS.append(entry)


def validate_output(path: str) -> None:
    """Refuse to overwrite an existing non-empty output file."""
    file_path = Path(path)
    if file_path.is_file() and file_path.stat().st_size > 0:
        raise ValueError(f"{file_path}: has existing content")
