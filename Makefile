.PHONY: setup notebook notebook-full test lint check api demo drift flow sync-public

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

demo:  ## Démarre l'API + l'IHM Streamlit de démonstration (ports 8000 et 8501)
	uv sync --all-extras
	uv run uvicorn churn_saas.api.main:app --host 0.0.0.0 --port 8000 &
	sleep 2
	uv run streamlit run demo/app.py --server.port 8501

drift:
	uv run python monitoring/drift_report.py

flow:
	uv run python flows/retrain_flow.py

sync-public:  ## Publie une version filtrée vers le dépôt public GitHub
	@bash scripts/sync_public.sh
