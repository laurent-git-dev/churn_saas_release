.PHONY: setup notebook notebook-full test test-slow lint check api drift flow

setup:
	uv sync --all-extras

notebook:
	uv run python notebooks/build_notebook.py

notebook-full:
	FORCE_RECALC=1 uv run python notebooks/build_notebook.py

# Tout le code Python du dépôt
CODE_PY := src/ tests/ flows/ monitoring/ notebooks/sections/ notebooks/build_notebook.py
CODE_TYPE := src/ flows/ monitoring/ notebooks/build_notebook.py

test:  ## Tests rapides ; les tests `slow` (gate qualité modèle) passent par `make test-slow`
	uv run pytest -q -m "not slow"

test-slow:
	uv run pytest -q -m slow

lint:
	uv run ruff check $(CODE_PY)
	uv run black --check $(CODE_PY)
	uv run mypy $(CODE_TYPE)

check: lint test
	$(MAKE) notebook

api:
	uv run uvicorn churn_saas.api.main:app --reload

drift:
	uv run python monitoring/drift_report.py

flow:
	uv run python flows/retrain_flow.py

