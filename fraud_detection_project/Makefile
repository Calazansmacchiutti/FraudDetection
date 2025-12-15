.PHONY: help install install-dev test lint format clean train api docker-build docker-run

# Default target
help:
	@echo "Fraud Detection Project"
	@echo ""
	@echo "Usage:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting"
	@echo "  make format       Format code"
	@echo "  make train        Train models"
	@echo "  make api          Start API server"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Run Docker container"
	@echo "  make clean        Clean up artifacts"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,notebooks]"

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

test-fast:
	pytest tests/ -v -x --tb=short

# Code quality
lint:
	flake8 src/ api/ tests/
	mypy src/ api/

format:
	black src/ api/ tests/ scripts/
	isort src/ api/ tests/ scripts/

check: lint test

# Training
train:
	python scripts/train.py

train-xgboost:
	python scripts/train.py --model xgboost

# API
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

api-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker-build:
	docker build -t fraud-detection:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models fraud-detection:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Notebooks
notebook:
	jupyter notebook notebooks/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name "*.egg-info" -delete
	rm -rf build/ dist/ htmlcov/ .coverage

clean-models:
	rm -rf models/*.pkl

clean-all: clean clean-models
	rm -rf data/processed/*
	rm -rf logs/*
	rm -rf reports/*
