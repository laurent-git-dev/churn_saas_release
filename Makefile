.PHONY: setup notebook notebook-full test lint check api drift flow

setup:
	uv sync --all-extras

notebook:
	uv run python notebooks/build_notebook.py

notebook-full:
	FORCE_RECALC=1 uv run python notebooks/build_notebook.py

test:
	uv run pytest -q

lint:
	uv run ruff check src/ tests/
	uv run black --check src/ tests/
	uv run mypy src/

check: lint test notebook

api:
	uv run uvicorn churn_saas.api.main:app --reload

drift:
	uv run python monitoring/drift_report.py

flow:
	uv run python flows/retrain_flow.py
