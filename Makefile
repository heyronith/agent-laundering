PYTHON ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

.PHONY: setup test prices smoke verify

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	$(PYTEST) -q

prices:
	$(PY) scripts/check_prices.py

smoke:
	$(PY) scripts/smoke_test.py

verify:
	$(PY) scripts/verify_logs.py
