.PHONY: install data train backtest test lint clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/build_dataset.py --year 2024

train:
	$(PYTHON) scripts/train_model.py

backtest:
	$(PYTHON) scripts/run_backtest.py --lap 30

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .cache .pytest_cache .ruff_cache
