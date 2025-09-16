SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
BLACK := .venv/bin/black
HEXNET := .venv/bin/hexnet
STREAMLIT := .venv/bin/streamlit

.PHONY: hexnets
hexnets: install

.PHONY: install
install: clean-venv
	mkdir -p figures/ runs/
	# this should ONLY ever be python3 and not python or .venv/bin/python
	@python3 -m venv .venv; \
	source .venv/bin/activate; \
	pip install --upgrade pip; \
	pip install -e .;

.PHONY: test
test:
	${PYTHON} -m unittest discover

.PHONY: run
run:
	@${HEXNET} sim -n 3 -lr 0.001 -t linear -e 200

.PHONY: run-streamlit
run-streamlit:
	@${STREAMLIT} run src/streamlit_app.py

.PHONY: ref-graphs
ref-graphs:
	@echo "Generating reference graphs..."
	@echo "----------------------------------------"
	@echo "n=2, layer_indices_terminal, multi_activation, structure_matplotlib"
	@${HEXNET} ref -n 2 -g layer_indices_terminal
	@${HEXNET} ref -n 2 -g multi_activation
	@${HEXNET} ref -n 2 -g structure_matplotlib

	@echo "----------------------------------------"
	@echo "n=3, layer_indices_terminal, multi_activation, structure_matplotlib"
	@${HEXNET} ref -n 3 -g layer_indices_terminal
	@${HEXNET} ref -n 3 -g multi_activation
	@${HEXNET} ref -n 3 -g structure_matplotlib

	@echo "----------------------------------------"
	@echo "n=4, layer_indices_terminal, multi_activation, structure_matplotlib"
	@${HEXNET} ref -n 4 -g layer_indices_terminal
	@${HEXNET} ref -n 4 -g multi_activation
	@${HEXNET} ref -n 4 -g structure_matplotlib

	@echo "----------------------------------------"
	@echo "n=5, layer_indices_terminal, multi_activation, structure_matplotlib"
	@${HEXNET} ref -n 5 -g layer_indices_terminal
	@${HEXNET} ref -n 5 -g multi_activation
	@${HEXNET} ref -n 5 -g structure_matplotlib

	@echo "----------------------------------------"
	@echo "n=6, layer_indices_terminal, multi_activation, structure_matplotlib"
	@${HEXNET} ref -n 6 -g layer_indices_terminal
	@${HEXNET} ref -n 6 -g multi_activation
	@${HEXNET} ref -n 6 -g structure_matplotlib

	@echo "----------------------------------------"
	@echo "n=7, layer_indices_terminal, multi_activation, structure_matplotlib"
	@${HEXNET} ref -n 7 -g layer_indices_terminal
	@${HEXNET} ref -n 7 -g multi_activation
	@${HEXNET} ref -n 7 -g structure_matplotlib

	@echo "----------------------------------------"
	@echo "n=8, layer_indices_terminal, multi_activation, structure_matplotlib"
	@${HEXNET} ref -n 8 -g layer_indices_terminal
	@${HEXNET} ref -n 8 -g multi_activation
	@${HEXNET} ref -n 8 -g structure_matplotlib


.PHONY: clean-all
clean-all: clean-venv clean-figures clean-runs

.PHONY: clean-venv
clean-venv:
	rm -rf .venv

.PHONY: clean-figures
clean-figures:
	rm -rf figures/*

.PHONY: clean-runs
clean-runs:
	rm -rf runs/*

.PHONY: lint-check
lint-check:
	black --check src -l 120

.PHONY: lint-format
lint-format:
	black src -l 120