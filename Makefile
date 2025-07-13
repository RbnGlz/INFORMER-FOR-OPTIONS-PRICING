# Makefile for Informer Option Pricing Project
# Professional automation for development workflows

.PHONY: help install install-dev clean test test-cov lint format security docs docker run-dev train predict profile benchmark deploy

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
POETRY := poetry
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := informer-option-pricing
DOCKER_IMAGE := $(PROJECT_NAME)
DOCKER_TAG := latest

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

## Help
help: ## Show this help message
	@echo "$(GREEN)Informer Option Pricing - Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

## Installation
install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[dev]
	pre-commit install

install-poetry: ## Install with Poetry
	@echo "$(GREEN)Installing with Poetry...$(NC)"
	$(POETRY) install --with dev

## Code Quality
lint: ## Run linting checks
	@echo "$(GREEN)Running linting checks...$(NC)"
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	mypy . --ignore-missing-imports || true

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	black .
	isort .

format-check: ## Check if code is formatted correctly
	@echo "$(GREEN)Checking code formatting...$(NC)"
	black --check --diff .
	isort --check-only --diff .

security: ## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	bandit -r . -f json -o bandit-report.json || true
	safety check || true

## Testing
test: ## Run tests
	@echo "$(GREEN)Running tests...$(NC)"
	pytest -v

test-cov: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest --cov=. --cov-report=xml --cov-report=html --cov-report=term-missing

test-fast: ## Run fast tests only
	@echo "$(GREEN)Running fast tests...$(NC)"
	pytest -v -m "not slow"

test-gpu: ## Run GPU-specific tests (requires GPU)
	@echo "$(GREEN)Running GPU tests...$(NC)"
	pytest tests/test_gpu.py -v || true

## Data and Training
create-sample-data: ## Create sample data for testing
	@echo "$(GREEN)Creating sample data...$(NC)"
	$(PYTHON) -c "import pandas as pd; import numpy as np; from datetime import datetime, timedelta; dates = pd.date_range(start='2023-01-01', periods=1000, freq='D'); data = {'fecha': dates, 'precio_subyacente': np.random.normal(100, 10, 1000), 'volatilidad_implicita': np.random.uniform(0.1, 0.5, 1000), 'tiempo_hasta_vencimiento': np.random.uniform(0.01, 1, 1000), 'precio_ejercicio': np.random.normal(100, 15, 1000), 'tipo_opcion': np.random.choice([0, 1], 1000), 'precio_opcion': np.random.uniform(1, 20, 1000)}; df = pd.DataFrame(data); df.to_csv('sample_option_data.csv', index=False); print('Sample data created')"

train: ## Train the model
	@echo "$(GREEN)Training the model...$(NC)"
	$(PYTHON) train.py

train-optimized: ## Train with optimized settings
	@echo "$(GREEN)Training with optimized settings...$(NC)"
	$(PYTHON) train_optimized.py

train-fast: ## Quick training for testing
	@echo "$(GREEN)Running fast training...$(NC)"
	$(PYTHON) train.py --epochs 2 --batch_size 16

predict: ## Make predictions
	@echo "$(GREEN)Making predictions...$(NC)"
	$(PYTHON) predict.py --run_id $(RUN_ID)

profile: ## Profile the model
	@echo "$(GREEN)Profiling the model...$(NC)"
	$(PYTHON) profile.py benchmark

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	$(PYTHON) profile.py benchmark
	$(PYTHON) profile.py cpu
	$(PYTHON) profile.py gpu || true

## Docker
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	$(DOCKER) build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-dev: ## Build development Docker image
	@echo "$(GREEN)Building development Docker image...$(NC)"
	$(DOCKER) build --target development -t $(DOCKER_IMAGE):dev .

docker-build-gpu: ## Build GPU Docker image
	@echo "$(GREEN)Building GPU Docker image...$(NC)"
	$(DOCKER) build --target gpu -t $(DOCKER_IMAGE):gpu .

docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(NC)"
	$(DOCKER) run --rm -it $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-dev: ## Run development Docker container
	@echo "$(GREEN)Running development Docker container...$(NC)"
	$(DOCKER) run --rm -it -p 8888:8888 -v $(PWD):/app $(DOCKER_IMAGE):dev

docker-run-gpu: ## Run GPU Docker container
	@echo "$(GREEN)Running GPU Docker container...$(NC)"
	$(DOCKER) run --rm -it --gpus all $(DOCKER_IMAGE):gpu

## Docker Compose
up: ## Start all services with docker-compose
	@echo "$(GREEN)Starting all services...$(NC)"
	$(DOCKER_COMPOSE) --profile dev up -d

up-train: ## Start training services
	@echo "$(GREEN)Starting training services...$(NC)"
	$(DOCKER_COMPOSE) --profile train up -d

up-gpu: ## Start GPU training services
	@echo "$(GREEN)Starting GPU training services...$(NC)"
	$(DOCKER_COMPOSE) --profile gpu up -d

up-api: ## Start API services
	@echo "$(GREEN)Starting API services...$(NC)"
	$(DOCKER_COMPOSE) --profile api up -d

down: ## Stop all services
	@echo "$(GREEN)Stopping all services...$(NC)"
	$(DOCKER_COMPOSE) down

logs: ## View logs
	@echo "$(GREEN)Viewing logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

## Documentation
docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation...$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-auto: ## Auto-rebuild documentation
	@echo "$(GREEN)Auto-rebuilding documentation...$(NC)"
	sphinx-autobuild docs/ docs/_build/html

## Monitoring
tensorboard: ## Start TensorBoard
	@echo "$(GREEN)Starting TensorBoard...$(NC)"
	tensorboard --logdir logs --host 0.0.0.0 --port 6006

mlflow: ## Start MLflow UI
	@echo "$(GREEN)Starting MLflow UI...$(NC)"
	mlflow ui --host 0.0.0.0 --port 5000

jupyter: ## Start Jupyter Lab
	@echo "$(GREEN)Starting Jupyter Lab...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

## Cleaning
clean: ## Clean up temporary files
	@echo "$(GREEN)Cleaning up temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage.*" -delete

clean-all: clean ## Clean everything including models and logs
	@echo "$(GREEN)Cleaning everything...$(NC)"
	rm -rf logs/
	rm -rf checkpoints/
	rm -rf experiments/
	rm -rf docs/_build/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -f *.pth
	rm -f *.csv
	rm -f *.png
	rm -f *.json

## Development
dev: install-dev create-sample-data ## Setup development environment
	@echo "$(GREEN)Development environment ready!$(NC)"

dev-docker: docker-build-dev ## Setup development environment with Docker
	@echo "$(GREEN)Development Docker environment ready!$(NC)"

pre-commit: ## Run pre-commit hooks
	@echo "$(GREEN)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

check: format-check lint security test ## Run all checks
	@echo "$(GREEN)All checks completed!$(NC)"

## CI/CD
ci-local: ## Simulate CI pipeline locally
	@echo "$(GREEN)Running CI pipeline locally...$(NC)"
	$(MAKE) clean
	$(MAKE) install-dev
	$(MAKE) check
	$(MAKE) test-cov
	$(MAKE) docker-build
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

release: ## Prepare for release
	@echo "$(GREEN)Preparing for release...$(NC)"
	$(MAKE) clean
	$(MAKE) check
	$(MAKE) test-cov
	$(MAKE) docs
	$(MAKE) docker-build
	@echo "$(GREEN)Release preparation completed!$(NC)"

## Monitoring and Profiling
monitor: ## Start monitoring stack
	@echo "$(GREEN)Starting monitoring stack...$(NC)"
	$(MAKE) tensorboard &
	$(MAKE) mlflow &
	@echo "$(GREEN)Monitoring stack started!$(NC)"

profile-memory: ## Profile memory usage
	@echo "$(GREEN)Profiling memory usage...$(NC)"
	$(PYTHON) -m memory_profiler train.py

profile-line: ## Profile line-by-line execution
	@echo "$(GREEN)Profiling line-by-line...$(NC)"
	kernprof -l -v train.py

## Deployment
deploy-staging: ## Deploy to staging
	@echo "$(GREEN)Deploying to staging...$(NC)"
	# Add your staging deployment commands here
	@echo "$(YELLOW)Staging deployment commands not implemented yet$(NC)"

deploy-prod: ## Deploy to production
	@echo "$(GREEN)Deploying to production...$(NC)"
	# Add your production deployment commands here
	@echo "$(YELLOW)Production deployment commands not implemented yet$(NC)"

## Utilities
setup: ## Complete project setup
	@echo "$(GREEN)Setting up project...$(NC)"
	$(MAKE) install-dev
	$(MAKE) create-sample-data
	$(MAKE) pre-commit
	@echo "$(GREEN)Project setup completed!$(NC)"

info: ## Show project information
	@echo "$(GREEN)Project Information:$(NC)"
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Pip version: $(shell $(PIP) --version)"
	@echo "Docker version: $(shell $(DOCKER) --version)"
	@echo "Docker Compose version: $(shell $(DOCKER_COMPOSE) --version)"
	@echo "Project name: $(PROJECT_NAME)"
	@echo "Docker image: $(DOCKER_IMAGE):$(DOCKER_TAG)"

# Environment-specific targets
.env:
	@echo "$(GREEN)Creating .env file...$(NC)"
	cp .env.example .env || echo "Please create a .env file manually"

# Target for running with different configs
train-config: ## Train with custom config
	@echo "$(GREEN)Training with config: $(CONFIG)$(NC)"
	$(PYTHON) train.py --config $(CONFIG)

# Database operations (if needed)
db-init: ## Initialize database
	@echo "$(GREEN)Initializing database...$(NC)"
	# Add database initialization commands here

db-migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	# Add database migration commands here

# Advanced Docker operations
docker-prune: ## Clean up Docker resources
	@echo "$(GREEN)Cleaning up Docker resources...$(NC)"
	$(DOCKER) system prune -f

docker-logs: ## View Docker logs
	@echo "$(GREEN)Viewing Docker logs...$(NC)"
	$(DOCKER) logs $(DOCKER_IMAGE):$(DOCKER_TAG)

# Performance testing
perf-test: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(NC)"
	pytest tests/test_performance.py -v

load-test: ## Run load tests
	@echo "$(GREEN)Running load tests...$(NC)"
	# Add load testing commands here

# Check if all required tools are installed
check-tools: ## Check if all required tools are installed
	@echo "$(GREEN)Checking required tools...$(NC)"
	@which $(PYTHON) > /dev/null || (echo "$(RED)Python not found$(NC)" && exit 1)
	@which $(PIP) > /dev/null || (echo "$(RED)Pip not found$(NC)" && exit 1)
	@which $(DOCKER) > /dev/null || (echo "$(RED)Docker not found$(NC)" && exit 1)
	@which $(DOCKER_COMPOSE) > /dev/null || (echo "$(RED)Docker Compose not found$(NC)" && exit 1)
	@echo "$(GREEN)All required tools are installed!$(NC)"