# GradEvo -- common tasks. Uses the active Python interpreter (activate your
# venv first, or override with `make PYTHON=/path/to/python <target>`).
PYTHON ?= python
ENV ?= LunarLanderContinuous-v3

.PHONY: help install experiments quick cartpole notebook test mlflow-ui clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install pinned dependencies and the gradevo package (editable)
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

experiments:  ## Full pre-registered run (N=20, 500k steps) on $(ENV) -- hours
	$(PYTHON) scripts/run_experiments.py --env $(ENV)

quick:  ## Fast reduced-N verification run on $(ENV) -- minutes
	$(PYTHON) scripts/run_experiments.py --env $(ENV) --quick

cartpole:  ## Quick secondary sanity-check run on CartPole-v1
	$(PYTHON) scripts/run_experiments.py --env CartPole-v1 --quick

notebook:  ## Regenerate the analysis notebook deterministically
	$(PYTHON) scripts/build_notebook.py

test:  ## Run the pytest suite
	$(PYTHON) -m pytest

mlflow-ui:  ## Launch the MLflow UI against the local SQLite store
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:  ## Remove caches and generated notebook checkpoints (keeps data/models)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
