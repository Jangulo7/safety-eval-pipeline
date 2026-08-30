.PHONY: help install doctor check dry-run run report gate ui test test-all lint format typecheck clean

PY ?= .venv/bin/python
PYTHONPATH := src

help:
	@echo "Safety Eval Pipeline — AISI Inspect safety benchmarks with a release gate"
	@echo
	@echo "  make install    Create .venv and install runtime + dev dependencies"
	@echo "  make doctor     Preflight: credentials, gated datasets, metric drift"
	@echo "  make check      Reproducibility gate: will these numbers be comparable?"
	@echo "  make dry-run    Print every cell that would run. Costs nothing."
	@echo "  make run        Run the matrix, render every report, then gate (exit 1 on breach)"
	@echo "  make report     Re-render reports from the latest results.json (no provider calls)"
	@echo "  make gate       Evaluate thresholds against the latest run; exit 1 on a breach"
	@echo "  make ui         Launch the Streamlit dashboard on :8501"
	@echo
	@echo "  make test       Unit + harness tests (no provider, no cost)"
	@echo "  make test-all   Also the integration tests (hits OpenRouter; costs money)"
	@echo "  make lint       ruff"
	@echo "  make typecheck  mypy"

install:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	@echo
	@echo "Now: cp .env.example .env and fill in OPENROUTER_API_KEY (and HF_TOKEN for XSTest)."
	@echo "Then: make doctor"

doctor:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m safety_eval.cli doctor --metrics

check:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m safety_eval.cli check

dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m safety_eval.cli run --dry-run

run:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m safety_eval.cli run --report
	PYTHONPATH=$(PYTHONPATH) $(PY) -m safety_eval.cli gate

report:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m safety_eval.cli report

gate:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m safety_eval.cli gate

ui:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -m "not integration"

test-all:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests app

format:
	$(PY) -m ruff format src tests app

typecheck:
	$(PY) -m mypy src/safety_eval --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage .ruff_cache
