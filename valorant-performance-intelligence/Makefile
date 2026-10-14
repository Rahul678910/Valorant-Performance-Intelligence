.PHONY: help install data train eda test dashboard docker-build docker-up clean

help:
	@echo "Valorant Performance Intelligence - Commands"
	@echo "  make install     : Install dependencies into active environment"
	@echo "  make data        : Run synthetic data synthesis and feature engineering"
	@echo "  make eda         : Generate exploratory charts and markdown report"
	@echo "  make train       : Train candidate models and serialize artifacts"
	@echo "  make test        : Run pytest unit test suite"
	@echo "  make dashboard   : Launch interactive Streamlit dashboard"
	@echo "  make docker-up   : Launch application via Docker Compose"
	@echo "  make clean       : Remove caches and temporary files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

data:
	python -m src.data_processing --n-samples 6000 --seed 42
	python -m src.feature_engineering

eda:
	python -m src.eda_analysis

train:
	python -m src.train_model
	python -m src.evaluate_model

test:
	pytest -v tests/

dashboard:
	streamlit run src/dashboard.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
