# ==============================================================================
# Makefile — tdigest-rs (core Rust + Python bindings)
# ==============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.SILENT:
MAKEFLAGS += --no-builtin-rules --no-print-directory
.DEFAULT_GOAL := help

CARGO ?= cargo
UV ?= uv

UV_CACHE_DIR ?= $(PWD)/.uv-cache
UV_ENV := UV_PROJECT_ENVIRONMENT=$(PWD)/.venv UV_CACHE_DIR=$(UV_CACHE_DIR)
PY_DIR := bindings/python
PY_TESTS := $(PY_DIR)/tests

.PHONY: help setup build build-rust build-python test rust-test py-test bench lint clean

help:
	@printf "\nTargets:\n"
	@printf "  %-16s %s\n" "setup" "Install Python deps for local dev"
	@printf "  %-16s %s\n" "build" "Build Rust crate and Python extension"
	@printf "  %-16s %s\n" "test" "Run Rust and Python API tests"
	@printf "  %-16s %s\n" "bench" "Run core benchmarks"
	@printf "  %-16s %s\n" "lint" "Run Rust fmt/clippy and Python ruff/mypy"
	@printf "  %-16s %s\n" "clean" "Remove local build artifacts"

setup:
	$(UV_ENV) $(UV) python install 3.12 || true
	(cd $(PY_DIR) && $(UV_ENV) $(UV) sync --all-groups)

build: build-rust build-python

build-rust:
	$(CARGO) build

build-python:
	(cd $(PY_DIR) && $(UV_ENV) $(UV) sync --all-groups)
	(cd $(PY_DIR) && $(UV_ENV) $(UV) run --no-sync maturin develop -F python)

test: rust-test py-test

rust-test:
	$(CARGO) test -- --quiet

py-test: build-python
	(cd $(PY_DIR) && $(UV_ENV) $(UV) run --no-sync pytest -q tests/test_api_python.py)

bench:
	$(CARGO) bench --bench tdigest_bench -- --noplot
	$(CARGO) bench --bench cdf_quantile_bench -- --noplot

lint:
	$(CARGO) fmt --all
	$(CARGO) clippy --all-targets --all-features -- -D warnings
	(cd $(PY_DIR) && $(UV_ENV) $(UV) run ruff check .)
	(cd $(PY_DIR) && $(UV_ENV) $(UV) run ruff format --check .)
	(cd $(PY_DIR) && $(UV_ENV) $(UV) run mypy tdigest_rs)

clean:
	rm -rf target .venv .uv-cache $(PY_DIR)/dist $(PY_DIR)/build $(PY_DIR)/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
