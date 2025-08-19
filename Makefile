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

.PHONY: clean-all
clean-all: clean-venv clean-figures

.PHONY: clean-venv
clean-venv:
	rm -rf .venv

.PHONY: clean-figures
clean-figures:
	rm -rf figures/*