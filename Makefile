PYTHON_MODULES := code
PYTHONPATH := .
VENV := venv
BIN := $(VENV)/bin

PYTHON := env PYTHONPATH=$(PYTHONPATH) $(BIN)/python
PIP := $(BIN)/pip

REQUIREMENTS := -r requirements.txt
PRE_COMMIT := $(BIN)/pre-commit

run:
	$(PYTHON) $(PYTHON_MODULES)/cross_validation.py

bootstrap: venv \
			requirements \
			pre-commit-hooks

venv:
	python -m venv $(VENV)

requirements:
	$(PIP) install $(REQUIREMENTS)

pre-commit-hooks:
	cp hooks/pre-commit .git/hooks/pre-commit
	$(PRE_COMMIT) install

test:
	venv/bin/pytest code/
