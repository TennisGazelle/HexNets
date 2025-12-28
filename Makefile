SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
BLACK := .venv/bin/black
HEXNET := .venv/bin/hexnet
STREAMLIT := .venv/bin/streamlit

.PHONY: hexnets
hexnets: install

.PHONY: install
install:
	@mkdir -p figures/ runs/
	# this should ONLY ever be python3 and not python or .venv/bin/python
	@python3 -m venv .venv; \
	source .venv/bin/activate; \
	pip install --upgrade pip; \
	pip install -e . ; \
	pip install -e .'[dev]' ;

.venv/: install

.PHONY: unit-test
unit-test:
	@${PYTHON} -m pytest tests

.PHONY: e2e-test
e2e-test:
	# clean up any previous runs
	rm -rf runs/e2etest-hex-train
	rm -rf runs/e2etest-mlp-train

	@echo '============================|E2E Tests|============================' ; \
	source .venv/bin/activate ; \

	echo '===> ref graph, hex, n=2, r=1' ; \
	hexnet ref -m hex -n 2 -r 1 -g structure_matplotlib ; \
	status_ref_hex=$$? ; \

	echo '===> ref graph, mlp, n=2,3,3,2' ; \
	hexnet ref -m mlp -g structure_matplotlib ; \
	status_ref_mlp=$$? ; \

	echo '===> train, hex, n=2, r=1, e=100' ; \
	hexnet train -m hex -n 2 -r 1 -e 100 -t identity -rn e2etest-hex-train ; \
	status_train_hex=$$? ; \

	echo '===> train, mlp, n=2,3,3,2, e=100' ; \
	hexnet train -m mlp -n 2 -e 100 -t identity -rn e2etest-mlp-train ; \
	status_train_mlp=$$? ; \
	
	echo '============================|E2E Tests|============================' ; \
	exit $$status_ref_hex || $$status_ref_mlp || $$status_train_hex || $$status_train_mlp


.PHONY: run
run:
	@${HEXNET} adhoc -n 3 -lr 0.001 -t linear -e 200

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
	@rm -rf .venv

.PHONY: clean-figures
clean-figures:
	@rm -rf figures/*

.PHONY: clean-runs
clean-runs:
	@rm -rf runs/*

.PHONY: lint-check
lint-check:
	@${BLACK} --check src -l 120

.PHONY: lint-format
lint-format:
	@${BLACK} src -l 120
