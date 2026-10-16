import json
from pathlib import Path
from typing import Any

from src.parser import FUNCTIONS, PROMPTS
from src.runner import DEFAULT_MODEL, get_model_object, get_system_prompt

MAX_STEPS = 200
MAX_NUMBER = 20
MAX_STRING = 200
NAME_HEAD = '{"name": "'
PARAMS_HEAD = '", "parameters": {'
JSON_END = "}}"
NUMBER_CHARS = "0123456789.eE+-"
NUMBER_DONE = {"zero", "int", "digits", "digits2"}
BOOL_WORDS = ("true", "false")

_NUMBER_TRANSITIONS: dict[str, dict[str, str]] = {
    "start": {"-": "-", "0": "zero", "digits": "int"},
    "-": {"0": "zero", "digits": "int"},
    "zero": {".": "frac", "e": "exp"},
    "int": {"digits": "int", ".": "frac", "e": "exp"},
    "frac": {"digits": "digits"},
    "digits": {"digits": "digits", "e": "exp"},
    "exp": {"+": "sign", "-": "sign", "digits": "digits2"},
    "sign": {"digits": "digits2"},
    "digits2": {"digits": "digits2"},
}


def _byte_decoder() -> dict[str, int]:
    """Return the vocab character -> byte mapping of byte-level BPE."""
    direct = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    table: dict[str, int] = {}
    extra = 0
    for byte in range(256):
        if byte in direct:
            table[chr(byte)] = byte
        else:
            table[chr(256 + extra)] = byte
            extra += 1
    return table


_BYTE_DECODER = _byte_decoder()


def _load_vocab(model: Any) -> dict[int, str]:
    """Return the vocabulary as token id -> text (byte exact).

    Read tokenizer.json: every supported tokenizer exposes it through
    get_path_to_tokenizer_file, unlike vocab.json, which some model
    repositories replace with tokenizer.model.
    """
    with open(model.get_path_to_tokenizer_file(), encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    vocab: dict[str, int] = data["model"]["vocab"]
    tokens: dict[int, str] = {}
    for token, token_id in vocab.items():
        raw = bytes(_BYTE_DECODER[char] for char in token)
        tokens[token_id] = raw.decode("utf-8", errors="surrogateescape")
    return tokens


def _string_safe(text: str) -> bool:
    """Return True if *text* needs no escaping inside a JSON string."""
    return not any(char in '"\\' or ord(char) < 0x20 for char in text)


def _char_class(char: str) -> str:
    """Return the number-DFA transition class of one character."""
    if char in "0123456789":
        return "digits"
    if char in "eE":
        return "e"
    return char


def _number_state(text: str) -> str | None:
    """Return the number-DFA state after *text*, or None if invalid."""
    state: str | None = "start"
    for char in text:
        if state is None:
            return None
        state = _NUMBER_TRANSITIONS[state].get(_char_class(char))
    return state


def _number_done(text: str) -> bool:
    """Return True if *text* is a complete JSON number."""
    return _number_state(text) in NUMBER_DONE


def _build_tables(model: Any) -> dict[str, Any]:
    """Load the vocabulary and precompute the per-state token tables."""
    tokens = _load_vocab(model)
    return {
        "tokens": tokens,
        "safe": [i for i, t in tokens.items() if t and _string_safe(t)],
        "numbers": [
            (i, t)
            for i, t in tokens.items()
            if t and all(char in NUMBER_CHARS for char in t)
        ],
        "close": [i for i, t in tokens.items() if t.startswith('"')],
        "comma": [i for i, t in tokens.items() if t.startswith(",")],
        "brace": [i for i, t in tokens.items() if t.startswith("}")],
        "name_masks": {},
    }


def _pick(model: Any, ids: list[int], allowed: list[int]) -> int:
    """Return the allowed token id with the highest logit."""
    logits: list[float] = model.get_logits_from_input_ids(ids)
    return max(allowed, key=logits.__getitem__)


def _force(model: Any, ids: list[int], fragment: str) -> None:
    """Append fixed text to the context without a model call."""
    ids.extend(model.encode(fragment).tolist()[0])


def _word_mask(
    tokens: dict[int, str], prefix: str, words: tuple[str, ...]
) -> list[int]:
    """Return the tokens that keep *prefix* on the way to one word."""
    return [
        i
        for i, t in tokens.items()
        if t and any(word.startswith(prefix + t) for word in words)
    ]


def _name_mask(
    tables: dict[str, Any], prefix: str, names: list[str]
) -> list[int]:
    """Return (cached) tokens that keep *prefix* on the way to a name."""
    cache: dict[str, list[int]] = tables["name_masks"]
    if prefix not in cache:
        cache[prefix] = _word_mask(tables["tokens"], prefix, tuple(names))
    return cache[prefix]


def _choose_name(
    model: Any, ids: list[int], tables: dict[str, Any], names: list[str]
) -> str:
    """Let the model choose one function name, token by token."""
    tokens: dict[int, str] = tables["tokens"]
    name = ""
    for _ in range(MAX_STEPS):
        if name in names:
            return name
        allowed = _name_mask(tables, name, names)
        if not allowed:
            raise ValueError(f"no function name continues {name!r}")
        token = _pick(model, ids, allowed)
        ids.append(token)
        name += tokens[token]
    raise ValueError("function name did not finish")


def _number_options(tables: dict[str, Any], text: str) -> list[int]:
    """Return the tokens that keep *text* a partial JSON number."""
    return [
        i for i, t in tables["numbers"] if _number_state(text + t) is not None
    ]


def _generate_number(
    model: Any, ids: list[int], tables: dict[str, Any], is_last: bool
) -> str:
    """Let the model generate one number value (always a float)."""
    tokens: dict[int, str] = tables["tokens"]
    text = ""
    for _ in range(MAX_STEPS):
        allowed = _number_options(tables, text)
        if not allowed:
            raise ValueError("cannot continue the number")
        logits: list[float] = model.get_logits_from_input_ids(ids)
        best = max(allowed, key=logits.__getitem__)
        stops = tables["brace"] if is_last else tables["comma"]
        wants_end = max(logits[i] for i in stops)
        finished = _number_done(text) and wants_end > logits[best]
        if finished or len(text) >= MAX_NUMBER:
            break
        ids.append(best)
        text += tokens[best]
    else:
        raise ValueError("number value did not finish")
    if "." not in text and "e" not in text and "E" not in text:
        _force(model, ids, ".0")
        text += ".0"
    return text


def _generate_bool(model: Any, ids: list[int], tables: dict[str, Any]) -> str:
    """Let the model choose one boolean value, token by token."""
    tokens: dict[int, str] = tables["tokens"]
    value = ""
    for _ in range(MAX_STEPS):
        if value in BOOL_WORDS:
            return value
        allowed = _word_mask(tokens, value, BOOL_WORDS)
        token = _pick(model, ids, allowed)
        ids.append(token)
        value += tokens[token]
    raise ValueError("boolean value did not finish")


def _generate_string(
    model: Any, ids: list[int], tables: dict[str, Any]
) -> str:
    """Let the model generate one string value, quotes included."""
    tokens: dict[int, str] = tables["tokens"]
    _force(model, ids, '"')
    text = '"'
    for _ in range(MAX_STEPS):
        logits: list[float] = model.get_logits_from_input_ids(ids)
        best = max(tables["safe"], key=logits.__getitem__)
        wants_close = max(logits[i] for i in tables["close"])
        if wants_close > logits[best] or len(text) >= MAX_STRING:
            break
        ids.append(best)
        text += tokens[best]
    else:
        raise ValueError("string value did not finish")
    _force(model, ids, '"')
    return text + '"'


def _generate_value(
    model: Any,
    ids: list[int],
    tables: dict[str, Any],
    kind: str,
    is_last: bool,
) -> str:
    """Let the model generate one value of the given schema type."""
    if kind == "number":
        return _generate_number(model, ids, tables, is_last)
    if kind == "boolean":
        return _generate_bool(model, ids, tables)
    return _generate_string(model, ids, tables)


def decode_function_call(
    model: Any, question: str, tables: dict[str, Any]
) -> dict[str, Any]:
    """Generate one function call for *question* with constrained decoding.

    The fixed structure is appended without model calls; the model only
    chooses the function name and the argument values.  The result is
    always valid JSON with exactly the keys and types of the function.
    """
    names = [fn["name"] for fn in FUNCTIONS]
    if not names:
        raise ValueError("no function definitions loaded")
    ids: list[int] = model.encode(get_system_prompt(question)).tolist()[0]
    parts: list[str] = []

    parts.append(NAME_HEAD)
    _force(model, ids, NAME_HEAD)
    name = _choose_name(model, ids, tables, names)
    parts.append(name)

    parts.append(PARAMS_HEAD)
    _force(model, ids, PARAMS_HEAD)

    function = next(fn for fn in FUNCTIONS if fn["name"] == name)
    parameters: dict[str, Any] = function["parameters"]
    keys = list(parameters)
    for index, key in enumerate(keys):
        if index:
            parts.append(", ")
            _force(model, ids, ", ")
        head = f'"{key}": '
        parts.append(head)
        _force(model, ids, head)
        kind = parameters[key]["type"]
        is_last = index + 1 == len(keys)
        parts.append(_generate_value(model, ids, tables, kind, is_last))

    parts.append(JSON_END)
    _force(model, ids, JSON_END)
    return _validate(question, "".join(parts))


def _validate(question: str, answer: str) -> dict[str, Any]:
    """Parse the generated text and build one output entry."""
    raw = answer.encode("utf-8", "surrogateescape")
    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON generated: {answer!r}") from exc
    return {
        "prompt": question,
        "name": parsed["name"],
        "parameters": parsed["parameters"],
    }


def run(output_path: str, model_name: str = DEFAULT_MODEL) -> None:
    """Decode every prompt and write the results to *output_path*."""
    if not FUNCTIONS or not PROMPTS:
        raise ValueError("input files were not parsed")
    print("loading model...", flush=True)
    model = get_model_object(model_name)
    tables = _build_tables(model)
    print("vocabulary ready", flush=True)
    results: list[dict[str, Any]] = []
    total = len(PROMPTS)
    for index, entry in enumerate(PROMPTS, start=1):
        question = entry["prompt"]
        print(f"[{index}/{total}] {question}", flush=True)
        result = decode_function_call(model, question, tables)
        print(f"    -> {result['name']} {result['parameters']}", flush=True)
        results.append(result)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {len(results)} calls to {output_path}", flush=True)
