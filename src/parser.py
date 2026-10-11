"""Parsing and validation of the project input files.

The two input files are described by pydantic models.  The loaders return
validated objects or raise InputError with a message suitable for the user.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class InputError(Exception):
    """A user-facing problem with one of the input files."""


class ParameterSpec(BaseModel):
    """Description of a single function parameter."""

    type: str


class ReturnSpec(BaseModel):
    """Description of a function return value."""

    type: str


class FunctionDefinition(BaseModel):
    """One function available to the model."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ReturnSpec


class Prompt(BaseModel):
    """One natural language request to process."""

    prompt: str


def _read_json(path: Path) -> Any:
    """Read a JSON document and report failures as InputError."""
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise InputError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON in {path}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc


def load_functions(path: str | Path) -> list[FunctionDefinition]:
    """Load and validate the function definitions file.

    Args:
        path: Location of the JSON file with function definitions.

    Returns:
        The validated definitions, in file order.

    Raises:
        InputError: If the file is missing, unreadable, or invalid.
    """
    file_path = Path(path)
    data = _read_json(file_path)
    if not isinstance(data, list):
        raise InputError(f"{file_path}: expected a JSON array")
    functions: list[FunctionDefinition] = []
    for index, item in enumerate(data):
        try:
            functions.append(FunctionDefinition.model_validate(item))
        except ValidationError as exc:
            raise InputError(
                f"{file_path}: invalid function at index {index}: {exc}"
            ) from exc
    names = [function.name for function in functions]
    if len(names) != len(set(names)):
        raise InputError(f"{file_path}: duplicate function names")
    return functions


def load_prompts(path: str | Path) -> list[Prompt]:
    """Load and validate the prompts file.

    Args:
        path: Location of the JSON file with prompts.

    Returns:
        The validated prompts, in file order.

    Raises:
        InputError: If the file is missing, unreadable, or invalid.
    """
    file_path = Path(path)
    data = _read_json(file_path)
    if not isinstance(data, list):
        raise InputError(f"{file_path}: expected a JSON array")
    prompts: list[Prompt] = []
    for index, item in enumerate(data):
        try:
            prompts.append(Prompt.model_validate(item))
        except ValidationError as exc:
            raise InputError(
                f"{file_path}: invalid prompt at index {index}: {exc}"
            ) from exc
    return prompts
