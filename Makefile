.PHONY: format static lint tests clean install help

help:
	@echo "Available targets:"
	@echo "  make format  - Format code with black"
	@echo "  make static  - Run static type checking with mypy"
	@echo "  make lint    - Run flake8 linter"
	@echo "  make tests   - Run pytest"
	@echo "  make install - Install dependencies"
	@echo "  make clean   - Remove cache files"

format:
	black src/ tests/

static:
	mypy src/ --ignore-missing-imports

lint:
	flake8 src/ tests/ --exclude=__init__.py --max-line-length=100

tests:
	pytest tests/ -v

tests-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
