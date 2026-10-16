install:
	@uv sync

run:
	@uv run python -m src

debug:
	@uv run python -m pdb -m src

clean:
	@find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -fr {} +
	@rm -fr .mypy_cache .ruff_cache .pytest_cache

# llm_sdk and moulinette are provided tools: exclude them from our lint.
lint:
	@uv run flake8 . --exclude=.venv,__pycache__,llm_sdk,moulinette
	@uv run mypy . --exclude '(llm_sdk|moulinette)' \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

lint-strict:
	@uv run flake8 . --exclude=.venv,__pycache__,llm_sdk,moulinette
	@uv run mypy . --exclude '(llm_sdk|moulinette)' --strict
