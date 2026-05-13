# ============================================================================
# HexNets Makefile
# ============================================================================
# This Makefile provides convenient targets for common development tasks.
#
# Quick Start:
#   make install          - Set up the virtual environment and install dependencies
#   make run              - Run a quick adhoc training example
#   make streamlit-run    - Launch the Streamlit web interface
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
# Documentation:
#   make pdf              - Build docs/latex/main.pdf via Docker (texlive-small)
#   make pdf-clean        - Remove the built PDF and LaTeX aux files
#
# Cleanup:
#   make clean-ref        - Remove all reference graph files
#   make clean-figures    - Remove all generated figure files
#   make clean-runs       - Remove all training run directories
#   make clean-venv       - Remove the virtual environment
#   make clean-all        - Remove everything (venv, figures, runs, refs, pdf)
#
# ============================================================================

SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
BLACK := .venv/bin/black
HEXNET := .venv/bin/hexnet
STREAMLIT := .venv/bin/streamlit

LATEX_DIR := docs/latex
LATEX_IMAGE := ghcr.io/xu-cheng/texlive-small:latest
LATEX_DOCKER := docker run --rm \
	-v "$(CURDIR)/$(LATEX_DIR):/workdir" -w /workdir \
	-u "$(shell id -u):$(shell id -g)" \
	$(LATEX_IMAGE)

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

.PHONY: build
build:
	@${PYTHON} -m build

.PHONY: stories-sync
stories-sync:
	@${PYTHON} scripts/sync_github_stories.py sync

.PHONY: unit-test
unit-test:
	@${PYTHON} -m pytest tests/

.PHONY: e2e-test
e2e-test:
	./e2e_test.sh E2E_EPOCHS=10

.PHONY: e2e-test-full
e2e-test-full:
	./e2e_test.sh E2E_EPOCHS=200
.PHONY: run
run:
	@${HEXNET} adhoc -n 3 -lr 0.001 -t linear_scale -e 200

.PHONY: streamlit-run
streamlit-run:
	@${STREAMLIT} run src/streamlit_app.py

.PHONY: streamlit-deploy
streamlit-deploy:
	@echo "============================================================================"
	@echo "Streamlit Cloud Deployment Preparation"
	@echo "============================================================================"
	@echo ""
	@echo "Checking deployment prerequisites..."
	@echo ""
	@# Check if app file exists
	@if [ ! -f "src/streamlit_app.py" ]; then \
		echo "ERROR: src/streamlit_app.py not found"; \
		exit 1; \
	else \
		echo "Streamlit app file found: src/streamlit_app.py"; \
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
	@echo "   - Main file path: src/streamlit_app.py"
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

.PHONY: pdf
pdf:
	@cp \
		reference/hexnet_n3_r0_structure.png \
		reference/hexnet_n3_multi_activation.png \
		docs/latex/
	@command -v docker >/dev/null || { echo "docker required for 'make pdf'"; exit 1; }
	@$(LATEX_DOCKER) latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	@echo "PDF written to $(LATEX_DIR)/main.pdf"

.PHONY: pdf-clean
pdf-clean:
	@command -v docker >/dev/null && $(LATEX_DOCKER) latexmk -C main.tex 2>/dev/null || true
	@rm -f $(LATEX_DIR)/main.pdf

.PHONY: clean-all
clean-all: clean-venv clean-figures clean-runs clean-ref pdf-clean

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

.PHONY: version-print
version-print:
	@$(PYTHON) scripts/bump_version.py pyproject.toml

.PHONY: version-bump-patch
version-bump-patch:
	@$(PYTHON) scripts/bump_version.py pyproject.toml --patch

.PHONY: version-bump-minor
version-bump-minor:
	@$(PYTHON) scripts/bump_version.py pyproject.toml --minor

.PHONY: version-bump-major
version-bump-major:
	@$(PYTHON) scripts/bump_version.py pyproject.toml --major