PYTHON ?= python
ROWS ?= 100000

install:
	$(PYTHON) -m pip install -r requirements.txt
run:
	$(PYTHON) -m nexus.cli run --rows $(ROWS)
generate:
	$(PYTHON) -m nexus.cli generate --rows $(ROWS)
pipeline:
	$(PYTHON) -m nexus.cli pipeline
train:
	$(PYTHON) -m nexus.cli train
api:
	uvicorn nexus.api:app --reload --port 8000
dashboard:
	streamlit run dashboard/app.py
test:
	pytest -q
lint:
	ruff check .
