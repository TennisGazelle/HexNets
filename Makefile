# ============================================================================
# HexNets Makefile
# ============================================================================
# This Makefile provides convenient targets for common development tasks.
#
# Quick Start:
#   make install          - Set up the virtual environment and install dependencies
#   make run              - Run a quick adhoc training example
#   make run-streamlit    - Launch the Streamlit web interface
#   make streamlit-deploy - Prepare and get instructions for deploying to Streamlit Cloud
#
# Testing:
#   make unit-test        - Run unit tests
#   make e2e-test         - Run end-to-end integration tests
#
# Backlog / issues:
#   make stories-sync     - Sync stories/*.md with GitHub issues (requires gh + PyYAML)
#
# Code Quality:
#   make lint             - Format code with Black (line length 120)
#   make lint-check       - Check code formatting without making changes (fails if formatting needed)
#
# Cleanup:
#   make clean-ref        - Remove all reference graph files
#   make clean-figures    - Remove all generated figure files
#   make clean-runs       - Remove all training run directories
#   make clean-venv       - Remove the virtual environment
#   make clean-all        - Remove everything (venv, figures, runs, refs)
#
# ============================================================================

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

.PHONY: stories-sync
stories-sync:
	@${PYTHON} scripts/sync_github_stories.py sync

.PHONY: unit-test
unit-test:
	@${PYTHON} -m pytest tests/

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
	@${STREAMLIT} run src/streamlit_main.py

.PHONY: streamlit-deploy
streamlit-deploy:
	@echo "============================================================================"
	@echo "Streamlit Cloud Deployment Preparation"
	@echo "============================================================================"
	@echo ""
	@echo "Checking deployment prerequisites..."
	@echo ""
	@# Check if app file exists
	@if [ ! -f "src/streamlit_main.py" ]; then \
		echo "ERROR: src/streamlit_main.py not found"; \
		exit 1; \
	else \
		echo "Streamlit app file found: src/streamlit_main.py"; \
	fi
	@# Check if requirements.txt exists
	@if [ ! -f "requirements.txt" ]; then \
		echo "WARNING: requirements.txt not found"; \
		echo "   Streamlit Cloud can use pyproject.toml, but requirements.txt is recommended"; \
		echo "   Create it with: pip freeze > requirements.txt"; \
	else \
		echo "requirements.txt found"; \
	fi
	@# Check if reference directory exists (for rotation comparison)
	@if [ ! -d "reference" ]; then \
		echo "WARNING: reference/ directory not found"; \
		echo "   Rotation Comparison tab will show warnings"; \
		echo "   Generate reference graphs with: hexnet ref --all"; \
	else \
		echo "reference/ directory found"; \
	fi
	@echo ""
	@echo "============================================================================"
	@echo "Deployment Instructions"
	@echo "============================================================================"
	@echo ""
	@echo "1. Ensure your code is pushed to a GitHub repository"
	@echo ""
	@echo "2. Go to https://share.streamlit.io/"
	@echo ""
	@echo "3. Sign in with your GitHub account"
	@echo ""
	@echo "4. Click 'New app' and select your repository"
	@echo ""
	@echo "5. Configure deployment:"
	@echo "   - Main file path: src/streamlit_main.py"
	@echo "   - Python version: 3.9 or higher"
	@echo "   - Requirements file: requirements.txt (or pyproject.toml)"
	@echo ""
	@echo "6. Click 'Deploy!'"
	@echo ""
	@echo "Note: Streamlit Cloud will automatically:"
	@echo "  - Install dependencies from requirements.txt or pyproject.toml"
	@echo "  - Run your app on their servers"
	@echo "  - Provide a public URL"
	@echo ""
	@echo "For more information, visit: https://docs.streamlit.io/streamlit-community-cloud"
	@echo ""
	@echo "============================================================================"

.PHONY: clean-all
clean-all: clean-venv clean-figures clean-runs clean-ref

.PHONY: clean-venv
clean-venv:
	@rm -rf .venv

.PHONY: clean-figures
clean-figures:
	@rm -rf figures/*

.PHONY: clean-runs
clean-runs:
	@rm -rf runs/*

.PHONY: clean-ref
clean-ref:
	@rm -rf reference/*.png

.PHONY: lint
lint:
	@${BLACK} src -l 120

.PHONY: lint-check
lint-check:
	@${BLACK} --check src -l 120 || (echo "Linting failed. Run 'make lint' to fix formatting issues." && exit 1)
