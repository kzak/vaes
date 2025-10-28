.PHONY: help format lint check test clean

help:
	@echo "Available commands:"
	@echo "  make format      - Format code with ruff"
	@echo "  make lint        - Lint code with ruff"
	@echo "  make check       - Run ruff checks without fixing"
	@echo "  make test        - Run tests with pytest"
	@echo "  make clean       - Clean cache and temporary files"

format:
	@echo "Formatting code with ruff..."
	uv run ruff format src/ notebooks/
	@echo "Formatting complete!"

lint:
	@echo "Linting and fixing code with ruff..."
	uv run ruff check --fix src/ notebooks/
	@echo "Linting complete!"

check:
	@echo "Checking code with ruff..."
	uv run ruff check src/ notebooks/
	uv run ruff format --check src/ notebooks/

test:
	@echo "Running tests..."
	uv run pytest tests/

clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete!"
