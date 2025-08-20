SHELL := /bin/bash
PYTHON := .venv/bin/python

hexnets: install

.PHONY: install
install:
	# this should ONLY ever be python3 and not python or .venv/bin/python
	@python3 -m venv .venv; \
	source .venv/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt

.PHONY: test
test:
	python -m unittest discover

.PHONY: run
run:
	@${PYTHON} working_hexnet.py

.PHONY: ref-graphs
ref-graphs:
	@${PYTHON} cli.py ref -n 2 -g activation
	@${PYTHON} cli.py ref -n 3 -g activation
	@${PYTHON} cli.py ref -n 4 -g activation
	@${PYTHON} cli.py ref -n 5 -g activation
	@${PYTHON} cli.py ref -n 6 -g activation
	@${PYTHON} cli.py ref -n 7 -g activation
	@${PYTHON} cli.py ref -n 8 -g activation


	@${PYTHON} cli.py ref -n 2 -g dot
	@${PYTHON} cli.py ref -n 3 -g dot
	@${PYTHON} cli.py ref -n 4 -g dot
	@${PYTHON} cli.py ref -n 5 -g dot
	@${PYTHON} cli.py ref -n 6 -g dot
	@${PYTHON} cli.py ref -n 7 -g dot
	@${PYTHON} cli.py ref -n 8 -g dot

.PHONY: clean-all
clean-all: clean-venv clean-figures

.PHONY: clean-venv
clean-venv:
	rm -rf .venv

.PHONY: clean-figures
clean-figures:
	rm -rf figures/*