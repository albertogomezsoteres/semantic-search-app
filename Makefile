PYTHON ?= .venv/bin/python
RUN = PYTHONPATH=src $(PYTHON) -m football_search

.PHONY: setup data transform validate index eval app test lint format all clean

setup:  ## Create the virtual env and install dev dependencies
	python3.11 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

data:  ## Download raw StatsBomb JSON (pinned commit) into data/raw
	$(RUN).extract

transform:  ## Flatten shots into data/staged/shots.parquet
	$(RUN).transform

validate:  ## Run data quality checks on the staged data
	$(RUN).validate

index:  ## Embed shots and build the Chroma index in chromadb/
	$(RUN).index

eval:  ## Score retrieval quality and write reports/
	$(RUN).evaluate

app:  ## Run the Gradio app locally
	$(PYTHON) app.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

all: data transform validate index eval  ## Rebuild everything from scratch

clean:  ## Remove staged data (raw downloads are kept)
	rm -rf data/staged
