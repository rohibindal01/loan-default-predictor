.PHONY: install hooks lint format typecheck test pipeline serve docker-up docker-down clean

install:
	poetry install --with dev

hooks:
	poetry run pre-commit install

lint:
	poetry run ruff check src tests
	poetry run ruff format --check src tests

format:
	poetry run ruff check --fix src tests
	poetry run ruff format src tests

typecheck:
	poetry run mypy src

test:
	poetry run pytest --cov=src --cov-report=term-missing --cov-report=xml

pipeline:
	poetry run dvc repro

serve:
	poetry run uvicorn src.serving.app:app --host 0.0.0.0 --port 8080 --reload

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
