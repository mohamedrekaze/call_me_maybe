from typing import Any

from src.parser import FUNCTIONS


def get_model_object() -> Any:
    """Create the LLM object (imported here to keep startup fast)."""
    from llm_sdk import Small_LLM_Model

    return Small_LLM_Model()


THINKING_OFF = " thinking\n\n\n\n"


def get_system_prompt(question: str = "", thinking_off: bool = True) -> str:
    """Build the complete chat prompt for one question.

    Returns the full prompt: system turn, user turn, and the assistant
    turn.  The empty thinking block (THINKING_OFF) is the exact suffix
    from the model's own chat template; it disables reasoning mode.
    """
    marker = THINKING_OFF if thinking_off else ""
    return (
        "<|im_start|>system\n"
        "You are a function-calling assistant."
        " Choose one function and give its arguments.\n"
        "Functions:\n"
        f"{get_functions_definition_prompt()}"
        "Answer with one JSON object and nothing else.\n"
        'Example: {"name": "fn_substitute_string_with_regex", '
        '"parameters": {"source_string": "hello world", '
        '"regex": "[aeiou]", "replacement": "*"}}'
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{marker}"
    )


def get_functions_definition_prompt() -> str:
    """Return the function definitions, one function per line."""
    lines: list[str] = []
    for func in FUNCTIONS:
        params = func["parameters"]
        params_line = ", ".join(
            f"{key}: {params[key]['type']}" for key in params
        )
        lines.append(
            f"{func['name']}({params_line})-> "
            f"{func['returns']['type']}: {func['description']}\n"
        )
    return "".join(lines)
