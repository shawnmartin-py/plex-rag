.DEFAULT_GOAL := help

.PHONY: help install chat web test test-unit test-integration test-e2e typecheck check pre-commit clear-history

help: ## Show this list of commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and the pre-commit hook
	uv sync
	pre-commit install

chat: ## Start an interactive recommendation session in the terminal
	plex-rag chat

web: ## Start the NiceGUI web UI at http://localhost:8080
	plex-rag-web

clear-history: ## Wipe the web UI's recent-conversations history
	plex-rag clear-history

test: ## Run the full test suite (unit + integration + e2e)
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit

test-integration: ## Run integration tests only
	pytest tests/integration

test-e2e: ## Run end-to-end tests only
	pytest tests/e2e

typecheck: ## Type-check with mypy (strict)
	mypy .

check: pre-commit test ## Run pre-commit hooks (lint, format, mypy) and tests — same gate as CI

pre-commit: ## Run all pre-commit hooks against every file
	pre-commit run --all-files
