*This project has been created as part of the 42 curriculum by morekaz.*

# call me maybe

## Description

A function-calling tool: it translates natural-language prompts into
structured function calls. Given `What is the sum of 2 and 3?`, the program
does not compute `5`; it produces the call that would:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers",
 "parameters": {"a": 2.0, "b": 3.0}}
```

The model is `Qwen/Qwen3-0.6B` (the subject default), run locally through the
provided `llm_sdk` wrapper. The core of the project is a **constrained
decoder**: instead of letting the model write free text and hoping it parses,
every generated token is restricted to the set of tokens that keep the output
on a path to a valid, schema-conforming JSON object. The output is therefore
100% valid JSON by construction, not by luck.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install        # same as: uv sync
```

`llm_sdk` is vendored next to `src/` and wired in as a uv workspace member, so
a plain `uv sync` is enough.

Run the program:

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>] [--model <model_name>]
```

Defaults: `data/input/functions_definition.json`,
`data/input/function_calling_tests.json`,
`data/output/function_calls.json`. The output file must not exist (or must be
empty); the program refuses to overwrite existing results.
`--model` selects the model and defaults to `Qwen/Qwen3-0.6B`;
`Qwen/Qwen3.5-0.8B` is also supported and tested.

Makefile targets: `install`, `run`, `debug` (pdb), `clean`, `lint`, `lint-strict`.

## Example usage

```bash
uv run python -m src
# or with explicit paths:
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
# or with the optional second model (fresh output path):
uv run python -m src --model Qwen/Qwen3.5-0.8B --output data/output/function_calls_qwen35.json
```

The run prints one line per prompt and writes the result array. For example:

```
[1/11] What is the sum of 2 and 3?
    -> fn_add_numbers {'a': 2.0, 'b': 3.0}
```

## Algorithm explanation

### Chat prompt

The model is prompted with its own chat format (`<|im_start|>` /
`<|im_end|>` turns): a system turn with the task instructions, the function
list (one line per function: name, parameter types, description) and one
worked example, then the user turn with the question. The assistant turn is
opened and immediately followed by the empty thinking block
(` thinking\n\n\n\n`) the exact no-think suffix rendered by the model's chat
template, which disables reasoning mode. The suffix was verified token by
token against `apply_chat_template(..., enable_thinking=False)`.

### Vocabulary

`tokenizer.json` is loaded once (`get_path_to_tokenizer_file`) and decoded
from byte-level (Byte Pair Encoding) BPE into plain text per token id. From it, a few tables are
precomputed:

- `safe`: tokens usable inside a JSON string (no `"`, `\`, control chars);
- `numbers`: tokens made only of `0123456789.eE+-`;
- `close` / `comma` / `brace`: tokens starting with `"`, `,`, `}`;
- `name_masks`: cache of allowed tokens per function-name prefix.

### Decoding loop

The fixed JSON structure is **appended without asking the model**:
`{"name": "`, `", "parameters": {`, each `"key": `, the separators and the
closing `}}`. The model is consulted only where a real choice exists:
the function name and the argument values. At every step only the masked
tokens compete:

1. **Function name**: allowed tokens are those whose text keeps the current
   prefix on the way to a known function name; the argmax over the mask is
   taken.
2. **Numbers**: a small (Deterministic Finite Automaton) DFA over JSON number syntax (optional minus, no
   leading zeros, at most one `.` and one `e`, digits after each) defines the
   allowed continuations. The number stops when it is complete and the best
   "stop token" (`,` for a non-last parameter, `}` for the last one) has a
   higher logit than the best continuation. Integers are normalized to floats
   (`.0` appended) because `number`-typed arguments must be Python floats.
3. **Strings**: the opening and closing quotes are forced. Content is chosen
   from `safe` tokens only, and stops when the best token starting with `"`
   beats the best safe token.
4. **Booleans**: prefix walk toward `true`/`false` (generic support; not used
   by the current test sets).

Stop detection deliberately compares against the **maximum logit over all
tokens starting with the delimiter**, not the single delimiter token, because
byte-level (Byte Pair Encoding) BPE merges common sequences (`",`, `}}`, and similar): the model usually
prefers a merged token when it wants to stop.

Finally the assembled text is parsed with `json.loads` as a safety net:
the state machine makes invalid JSON impossible, and the parse proves it.

## Design decisions

- **Forced scaffolding instead of model output**: the JSON skeleton cannot be
  corrupted because the model never chooses it. This is what makes the output
  valid regardless of model quality.
- **Mask, then argmax**: the model still decides *everything that matters*
  (which function, which argument values) using its own logits; no heuristics
  pick functions or values.
- **Float normalization**: the graded functions assert `isinstance(x, float)`;
  an integer literal fails. Appending `.0` when needed keeps the JSON type
  `number` while satisfying Python's type checks.
- **`tokenizer.json` over `vocab.json`**: newer Qwen tokenizers (e.g.
  `Qwen/Qwen3.5-0.8B`) declare `vocab_file = tokenizer.model`, which does not
  exist in the repository, so the SDK's vocab accessor 404s. Reading
  `tokenizer.json` works for both model generations and keeps the project
  portable across models.
- **Only public SDK methods** are used (`encode`, `decode`,
  `get_logits_from_input_ids`, `get_path_to_tokenizer_file`), as the subject
  requires.
- **Pydantic** models validate both input files; malformed or missing files
  produce a clear error message and exit code 1 instead of a traceback.
- **uv workspace** with `llm_sdk` as a member keeps the vendored SDK
  importable after a plain `uv sync`.

## Performance analysis

Measured on the public set (11 prompts), 8-core CPU:

| model | wall time | score |
|---|---|---|
| `Qwen/Qwen3-0.6B` (default) | ~2m20s | 10/11 (90.9%) |
| `Qwen/Qwen3.5-0.8B` (optional) | ~3m40s | 10/11 (90.9%) |

- **Accuracy**: 90.9% correct function selection and argument extraction on
  the public set. The single remaining miss is a regex case where the model
  emits `aeiou` (a literal sequence) instead of a character class such as
  `[aeiou]`; the result of `re.sub` differs from the expected output.
- **Validity**: 100% of outputs parse as JSON with exactly the three keys
  `prompt`, `name`, `parameters` and the parameter types of the chosen
  function, guaranteed by construction and re-checked with `json.loads`.
- **Speed**: well under the 5-minute budget. The decoder spends one forward
  pass per generated token (forced fragments cost none); a typical call needs
  9-40 passes. Wall time scales with machine load: the same run measured
  2m20s to 5m58s depending on ambient CPU load.

## Challenges faced

- **Thinking mode**: without the empty thinking block the model "reasons" for
  over a hundred tokens instead of answering. The fix was to copy the exact
  no-think suffix from the model's own chat template; spelling matters to the
  token, so it is verified by token ids in tests.
- **Runaway strings**: comparing the stop decision against the bare `"` token
  missed the merged tokens the model actually prefers, so strings never
  closed. Taking the max over all tokens starting with `"` (same for `,`/`}`)
  fixed it.
- **Types**: JSON integers versus Python floats (`assert isinstance`),
  leading zeros being invalid JSON (`01`), and floats needing `.0`.
- **Portability**: `get_path_to_vocab_file()` 404s on Qwen3.5 tokenizers; the
  decoder now reads `tokenizer.json` instead. The no-think suffix was
  re-verified against both models' templates.
- **Timing under load**: the per-forward cost varies with CPU load; the
  decoder keeps the number of forward passes minimal (one per emitted token).

## Testing strategy

- **End-to-end checks**: the produced `data/output/function_calls.json` was
  compared against the expected calls and outputs for the public set; the
  scores above come from that comparison.
- **Tokenizer/template checks**: the no-think suffix is compared token by token
  with `apply_chat_template(..., enable_thinking=False)` for every supported
  model; the vocabulary loaded from `tokenizer.json` is checked against
  `vocab.json`.
- **Smoke tests**: small scripts (`test.py`, `test/test.py`) print token ids,
  logits and greedy continuations for individual prompts to verify the
  pipeline before full runs.
- **Output validation**: every run's file is parsed and checked for exactly
  `prompt` / `name` / `parameters` per object.
- **Lint**: `make lint` (flake8 + mypy with the subject's mandatory flags) is
  green; `lint-strict` is available.

## Resources

- [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Transformers documentation](https://huggingface.co/docs/transformers)
- [Tokenizers documentation](https://huggingface.co/docs/tokenizers)
- [JSON specification, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)
- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909)
- [Pydantic documentation](https://docs.pydantic.dev)
- [Python `json` module](https://docs.python.org/3/library/json.html)

### How AI was used

AI assistance was used to explain and demonstrate concepts: how byte pair 
encoding turns text into tokens, how the chat template and its thinking 
markers control generation, how constrained decoding with logit masks guarantees
valid JSON, and how a small state machine enforces JSON number syntax. All code,
design decisions and testing were done by the author.
