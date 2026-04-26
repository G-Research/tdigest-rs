# ==============================================================================
# Makefile — tdigest-rs (Rust core + Python extension + Java/JNI)
# ==============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.SILENT:
MAKEFLAGS += --no-builtin-rules --no-print-directory
.DEFAULT_GOAL := help

# ----------------------------------------------------------------------
# Pretty output helpers
# ----------------------------------------------------------------------
STYLE_OK    := $(shell tput setaf 2 2>/dev/null || printf '\033[32m')
STYLE_ERR   := $(shell tput setaf 1 2>/dev/null || printf '\033[31m')
STYLE_BOLD  := $(shell tput bold 2>/dev/null     || printf '\033[1m')
STYLE_RESET := $(shell tput sgr0 2>/dev/null     || printf '\033[0m')
STYLE_CODE  := $(shell tput setaf 6 2>/dev/null || printf '\033[36m')

define banner
	@printf "\n$(STYLE_BOLD)==> %s$(STYLE_RESET)\n" "$(1)"
endef
define need
	@command -v $(1) >/dev/null 2>&1 || { printf "$(STYLE_ERR)✗ Missing dependency: $(1)$(STYLE_RESET)\n"; exit 1; }
	@printf "$(STYLE_OK)✓ $(1)$(STYLE_RESET)\n"
endef

# ----------------------------------------------------------------------
# Project naming
# ----------------------------------------------------------------------
CRATE_NAME      ?= tdigest_rs
PROJECT_SLUG    ?= tdigest-rs
JAR_ARTIFACT    ?= $(PROJECT_SLUG)

# ----------------------------------------------------------------------
# Tools & paths
# ----------------------------------------------------------------------
PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)

CARGO  ?= cargo
UV     ?= uv

# Force uv to use the repo-root venv
UV_CACHE_DIR ?= $(PWD)/.uv-cache
UV_ENV := UV_PROJECT_ENVIRONMENT=$(PWD)/.venv UV_CACHE_DIR=$(UV_CACHE_DIR)

# Python layout
PY_DIR          := bindings/python
PY_PKG_DIR      := $(PY_DIR)/tdigest_rs
PY_TESTS_DIR    := $(PY_DIR)/tests

# Upstream tdigest-rs review compatibility snapshot
COMPAT_ROOT        := compat/tdigest-rs-upstream
COMPAT_PY_DIR      := $(COMPAT_ROOT)/bindings/python
COMPAT_TESTS_DIR   := $(COMPAT_PY_DIR)/tests
COMPAT_BENCH       := $(COMPAT_PY_DIR)/benchmarks/run.py
COMPAT_ADAPTER_DIR := $(COMPAT_ROOT)/python

# Java layout (wrapper-only — no system Gradle fallback)
JAVA_SRC        := bindings/java
JAVA_BUILD_REL  := build
JAVA_LIBS_REL   := $(JAVA_BUILD_REL)/libs
GRADLEW_PATH    := $(JAVA_SRC)/gradlew
GRADLE          := $(GRADLEW_PATH)
GRADLE_USER_HOME ?= $(PWD)/.gradle-user-home

# Integration tests
INTEG_DIR       := integration/api_coherence
INTEG_TESTS_DIR := $(INTEG_DIR)

# Build profile (dev by default). Release only in release-* targets.
PROFILE ?= dev
ifeq ($(PROFILE),release)
  CARGO_PROFILE_FLAG := --release
  CARGO_DIR := release
else
  CARGO_PROFILE_FLAG :=
  CARGO_DIR := debug
endif
LIB_DIR  := target/$(CARGO_DIR)

# Version (from Cargo.toml)
VER := $(shell sed -n 's/^version\s*=\s*"\(.*\)"/\1/p' Cargo.toml | head -1)

# Platform detection → shared library basename
UNAME_S := $(shell uname -s | tr '[:upper:]' '[:lower:]')
UNAME_M := $(shell uname -m)
ifeq ($(findstring linux,$(UNAME_S)),linux)
  PLAT := linux
  LIBBASENAME := lib$(CRATE_NAME).so
else ifeq ($(findstring darwin,$(UNAME_S)),darwin)
  PLAT := macos
  LIBBASENAME := lib$(CRATE_NAME).dylib
else ifeq ($(findstring mingw,$(UNAME_S))$(findstring msys,$(UNAME_S)),mingwmsys)
  PLAT := windows
  LIBBASENAME := $(CRATE_NAME).dll
else
  PLAT := linux
  LIBBASENAME := lib$(CRATE_NAME).so
endif

ifeq ($(UNAME_M),x86_64)
  ARCH := x86_64
else ifeq ($(UNAME_M),amd64)
  ARCH := x86_64
else ifeq ($(UNAME_M),aarch64)
  ARCH := aarch64
else ifeq ($(UNAME_M),arm64)
  ARCH := aarch64
else ifneq (,$(filter i386 i686 x86,$(UNAME_M)))
  ARCH := x86
else
  ARCH := $(UNAME_M)
endif

# CLI binary (leave as-is if unchanged)
CLI_BIN  ?= tdigest
CLI_PATH := $(LIB_DIR)/$(CLI_BIN)

# Python distributions
DIST      ?= dist
DIST_DEV  ?= dist-dev
PUBLISH_DRY_RUN ?= 0

# Canonical API JAR path (symlink updated by release-jar)
API_JAR := $(DIST)/$(JAR_ARTIFACT)-latest.jar

# ==============================================================================
# Guards
# ==============================================================================
.PHONY: JAVA_WRAPPER_CHECK
JAVA_WRAPPER_CHECK:
	@test -x "$(GRADLEW_PATH)" || { \
	  echo "$(STYLE_ERR)✗ Missing Gradle wrapper ($(GRADLEW_PATH))$(STYLE_RESET)"; \
	  echo "  Bootstrap once: (cd $(JAVA_SRC) && gradle wrapper --gradle-version 9.1.0 --distribution-type bin)"; \
	  exit 1; }

# ==============================================================================
# PHONY TARGETS
# ==============================================================================
.PHONY: help setup clean \
        build build-rust build-python build-java \
        test rust-test py-test java-test compat-test compat-bench naming-hygiene \
        lint setup-hooks \
        smoke-rust-cli smoke-wheel  \
        release-rust release-wheel release-jar release \
        publish-check publish-pypi publish-cargo publish-maven publish

# ==============================================================================
# Setup
# ==============================================================================
setup:
	$(call banner,Checking required host tools)
	$(call need,rustup)
	$(call need,cargo)
	$(call need,git)
	$(call need,uv)
	@test -x "$(GRADLEW_PATH)" || { \
	  echo "$(STYLE_ERR)✗ Missing Gradle wrapper ($(GRADLEW_PATH))$(STYLE_RESET)"; \
	  echo "  Bootstrap once: (cd $(JAVA_SRC) && gradle wrapper --gradle-version 9.1.0 --distribution-type bin)"; \
	  exit 1; }
	$(UV) --version
	$(call banner,Create repo .venv and install Python deps (all groups))
	$(UV_ENV) $(UV) python install 3.12 || true
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) sync --all-groups)
	$(call banner,Quick Python import smoke)
	$(UV_ENV) $(UV) run python -c "import polars as pl, sys; print('python', sys.version.split()[0], '| polars', pl.__version__, '| env ok')"
	$(MAKE) setup-hooks

setup-hooks:
	$(call banner,Install pre-commit hooks with repo-local cache)
	mkdir -p .pre-commit-cache
	mkdir -p .uv-cache
	@if [ ! -x "$(PWD)/.venv/bin/python" ]; then \
	  echo "$(STYLE_ERR)✗ Missing .venv Python at $(PWD)/.venv/bin/python$(STYLE_RESET)"; \
	  echo "  Run: make setup"; \
	  exit 1; \
	fi
	@if ! "$(PWD)/.venv/bin/python" -m pre_commit --version >/dev/null 2>&1; then \
	  echo "pre_commit module missing in .venv; syncing Python deps from $(PY_DIR)"; \
	  (cd "$(PY_DIR)" && $(UV_ENV) $(UV) sync --all-groups); \
	fi
	PRE_COMMIT_HOME="$(PWD)/.pre-commit-cache" UV_CACHE_DIR="$(PWD)/.uv-cache" "$(PWD)/.venv/bin/python" -m pre_commit install
	@HOOK=".git/hooks/pre-commit"; \
	if [ ! -f "$$HOOK" ]; then \
	  echo "$(STYLE_ERR)✗ pre-commit hook not found at $$HOOK$(STYLE_RESET)"; \
	  exit 1; \
	fi; \
	if ! grep -q 'PRE_COMMIT_HOME' "$$HOOK" || ! grep -q 'UV_CACHE_DIR' "$$HOOK"; then \
	  TMP_HOOK="$$(mktemp)"; \
	  awk '/^export PRE_COMMIT_HOME=/ {next} /^export UV_CACHE_DIR=/ {next} 1; /# end templated/ {print "export PRE_COMMIT_HOME=\"$${PRE_COMMIT_HOME:-$(PWD)/.pre-commit-cache}\""; print "export UV_CACHE_DIR=\"$${UV_CACHE_DIR:-$(PWD)/.uv-cache}\""}' "$$HOOK" > "$$TMP_HOOK"; \
	  mv "$$TMP_HOOK" "$$HOOK"; \
	  chmod +x "$$HOOK"; \
	fi
	@printf "$(STYLE_OK)✓ pre-commit installed (PRE_COMMIT_HOME=.pre-commit-cache, UV_CACHE_DIR=.uv-cache)$(STYLE_RESET)\n"

# ==============================================================================
# Build (dev by default)
# ==============================================================================
build: build-rust build-python build-java
	@printf "$(STYLE_OK)✓ all components built (PROFILE=$(PROFILE))$(STYLE_RESET)\n"

build-rust:
	$(call banner,Build: Rust lib + CLI ($(PROFILE)))
	$(CARGO) build $(CARGO_PROFILE_FLAG)
	$(CARGO) build $(CARGO_PROFILE_FLAG) --bin $(CLI_BIN)

# Always run maturin from bindings/python where its pyproject.toml lives
build-python:
	$(call banner,Build: Python extension (maturin develop, dev))
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) run --no-sync maturin develop -F python)

build-java: JAVA_WRAPPER_CHECK
	$(call banner,Build: Java via Gradle (clean + jar))
	@GRADLE_USER_HOME="$(GRADLE_USER_HOME)" "$(GRADLE)" --no-daemon --console=plain -p "$(JAVA_SRC)" clean jar
	@LATEST="$$(ls -1t "$(JAVA_SRC)/$(JAVA_LIBS_REL)/$(JAR_ARTIFACT)-"*.jar 2>/dev/null | head -1)"; \
	[ -n "$$LATEST" ] || { echo "$(STYLE_ERR)✗ No JAR produced$(STYLE_RESET)"; exit 1; }; \
	printf "$(STYLE_OK)✓ Built %s$(STYLE_RESET)\n" "$$LATEST"

# ==============================================================================
# Tests
# ==============================================================================
.PHONY: test rust-test py-test java-test compat-test compat-bench naming-hygiene

test: naming-hygiene rust-test java-test py-test
	@echo "✅ all unit tests passed"

naming-hygiene:
	$(call banner,Naming hygiene: reject legacy identifiers)
	@! rg -n --hidden --glob '!.git' --glob '!.git/**' --glob '!target/**' '(gr[-_]tdigest|_gr[_]tdigest|ingolfured/gr[-]tdigest|gr[.]tdigest)' .

rust-test:
	$(CARGO) test -- --quiet

# Explicit path to bindings/python/tests (pytest discovers from pyproject too)
py-test: build-python
	$(UV_ENV) $(UV) run --project "$(PY_DIR)" --no-sync python -m pytest -q "$(PY_TESTS_DIR)" "$(INTEG_TESTS_DIR)"

compat-test:
	PYTHONPATH="$(PWD)/$(COMPAT_ADAPTER_DIR):$(PWD)/$(COMPAT_PY_DIR):$${PYTHONPATH:-}" $(UV_ENV) $(UV) run --project "$(PY_DIR)" --no-sync python -m pytest -q "$(COMPAT_TESTS_DIR)"

compat-bench:
	PYTHONPATH="$(PWD)/$(COMPAT_ADAPTER_DIR):$(PWD)/$(COMPAT_PY_DIR):$${PYTHONPATH:-}" $(UV_ENV) $(UV) run --project "$(PY_DIR)" --no-sync python "$(COMPAT_BENCH)"

java-test: JAVA_WRAPPER_CHECK
	@GRADLE_USER_HOME="$(GRADLE_USER_HOME)" "$(GRADLE)" --no-daemon --console=plain -p "$(JAVA_SRC)" test

# ==============================================================================
# Lint (autofix where possible) + docs must be warning-free
# ==============================================================================
lint:
	$(call banner,Naming hygiene: reject legacy identifiers)
	@$(MAKE) naming-hygiene

	$(call banner,Format Rust)
	$(CARGO) fmt --all

	$(call banner,Rust: clippy --fix (may modify files))
	$(CARGO) clippy --fix --allow-dirty --allow-staged

	$(call banner,Python: ruff check --fix + ruff format)
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) run ruff check --fix .)
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) run ruff format .)

	$(call banner,Python: mypy (auto-install types if missing))
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) run python -m mypy tdigest_rs --install-types --non-interactive)

	$(call banner,Rust docs: deny warnings)
	@set -euo pipefail; \
	export RUSTDOCFLAGS="-D warnings -D rustdoc::private_intra_doc_links -D rustdoc::broken_intra_doc_links"; \
	$(CARGO) doc --no-deps

# ==============================================================================
# Clean (single target)
# ==============================================================================
clean:
	$(call banner,Clean: Rust target/ and cargo)
	rm -rf target/ || true
	$(CARGO) clean || true
	$(call banner,Clean: Gradle build outputs)
	@{ [ -x "$(GRADLEW_PATH)" ] && GRADLE_USER_HOME="$(GRADLE_USER_HOME)" "$(GRADLE)" --no-daemon -p "$(JAVA_SRC)" clean || true; }
	rm -rf "$(JAVA_SRC)/$(JAVA_BUILD_REL)" || true
	$(call banner,Clean: Python build artifacts)
	@find "$(PY_DIR)" -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	rm -f  "$(PY_PKG_DIR)"/*.so "$(PY_PKG_DIR)"/*.dylib "$(PY_PKG_DIR)"/*.pyd "$(PY_PKG_DIR)"/*.dll || true
	rm -rf "$(PY_DIR)/build/" "$(PY_DIR)"/*.egg-info/ || true
	$(call banner,Clean: Wheel dists)
	rm -rf "$(DIST)" "$(DIST_DEV)" || true
	@printf "$(STYLE_OK)✓ cleaned all artifacts$(STYLE_RESET)\n"

# ==============================================================================
# Smoke tests
# ==============================================================================
Q ?= 0.5

smoke-rust-cli:
	@set -eu; \
	BIN="target/$(CARGO_DIR)/$(CLI_BIN)"; \
	[ -x "$$BIN" ] || { echo "❌ missing CLI at $$BIN; build first (PROFILE=$(PROFILE))"; exit 1; }; \
	OUT="$$( echo '0 1 2 3' | "$$BIN" quantile --stdin --p $(Q) --no-header --output csv | cut -d, -f2 )"; \
	printf "CLI p%.3g -> %s\n" "$(Q)" "$$OUT"; \
	[ "$$OUT" = "1.5" ] || { echo "❌ CLI quantile mismatch (got '$$OUT', want '1.5')"; exit 1; }; \
	echo "✅ smoke-rust-cli ok (PROFILE=$(PROFILE))"

smoke-wheel:
	@set -eu; \
	LATEST="$$(ls -1t "$(DIST)"/*.whl "$(DIST_DEV)"/*.whl 2>/dev/null | head -1 || true)"; \
	[ -n "$$LATEST" ] || { echo "❌ no wheel found in $(DIST)/ or $(DIST_DEV)/"; exit 1; }; \
	TMP="$$(mktemp -d)"; \
	$(UV) venv --python 3.12 --seed "$$TMP"; \
	$(UV) pip install --python "$$TMP/bin/python" --upgrade pip; \
	$(UV) pip install --python "$$TMP/bin/python" 'polars>=1.34.0,<2.0.0' numpy; \
	$(UV) pip install --python "$$TMP/bin/python" --no-index --find-links "$$(dirname "$$LATEST")" tdigest-rs; \
	"$$TMP/bin/python" -c "import tdigest_rs as td; d=td.TDigest.from_array([0.0,1.0,2.0,3.0], max_size=100, scale='k2'); print('p50=', d.quantile(0.5)); print('cdf=', td.cdf); print('ok')"; \
	rm -rf "$$TMP"; \
	echo "✅ smoke-wheel ok (file=$$LATEST)"

# ==============================================================================
# Releases (release profile only)
# ==============================================================================
release-rust:
	$(call banner,Release: Rust CLI (release) + smoke)
	$(MAKE) build-rust PROFILE=release
	$(MAKE) smoke-rust-cli PROFILE=release
	@printf "• CLI binary : %s\n" "target/release/$(CLI_BIN)"

release-wheel:
	$(call banner,Release: Python PROD wheel (manylinux_2_28 + zig) + smoke)
	mkdir -p "$(DIST)"
	@(cd "$(PY_DIR)" && $(UV_ENV) $(UV) run maturin build -r -F python --compatibility manylinux_2_28 --zig -o "$(abspath $(DIST))")
	@printf "$(STYLE_OK)✓ wheel(s) in $(DIST)$(STYLE_RESET)\n"
	$(MAKE) -f $(CURDIR)/Makefile smoke-wheel

release-jar: JAVA_WRAPPER_CHECK
	$(call banner,Release: Java (Gradle) + publish to dist)
	@GRADLE_USER_HOME="$(GRADLE_USER_HOME)" "$(GRADLE)" --no-daemon -p "$(JAVA_SRC)" clean smoke
	mkdir -p "$(DIST)"
	LATEST="$(JAVA_SRC)/$(JAVA_LIBS_REL)/$(JAR_ARTIFACT)-$(VER).jar"
	[ -f "$$LATEST" ]
	cp -v "$$LATEST" "$(DIST)/"
	BASENAME="$$(basename "$$LATEST")"
	ln -sfn "$$BASENAME" "$(API_JAR)"
	echo "• Published: $(DIST)/$$BASENAME"
	echo "• Symlink : $(API_JAR) -> $$BASENAME"

release: release-rust release-wheel release-jar
	@printf "\n$(STYLE_BOLD)==> Release complete$(STYLE_RESET)\n"
	@printf "  CLI binary : %s\n" "target/release/$(CLI_BIN)"
	@printf "  Wheels dir : %s\n" "$(DIST)"
	@printf "  Java JAR   : %s (symlink)\n" "$(API_JAR)"
	@printf "$(STYLE_OK)✓ All release artifacts built & smoke-tested$(STYLE_RESET)\n"

publish-check:
	$(call banner,Publish: validate environment)
	@set -eu; \
	if [ "$(PUBLISH_DRY_RUN)" = "1" ]; then \
	  printf "$(STYLE_OK)✓ dry-run mode enabled (PUBLISH_DRY_RUN=1)$(STYLE_RESET)\n"; \
	  exit 0; \
	fi; \
	[ -n "$${MATURIN_PYPI_TOKEN:-}" ] || { echo "$(STYLE_ERR)✗ Missing MATURIN_PYPI_TOKEN$(STYLE_RESET)"; exit 1; }; \
	[ -n "$${CARGO_REGISTRY_TOKEN:-}" ] || { echo "$(STYLE_ERR)✗ Missing CARGO_REGISTRY_TOKEN$(STYLE_RESET)"; exit 1; }; \
	[ -n "$${MAVEN_REPOSITORY_URL:-}" ] || { echo "$(STYLE_ERR)✗ Missing MAVEN_REPOSITORY_URL$(STYLE_RESET)"; exit 1; }; \
	[ -n "$${MAVEN_USERNAME:-}" ] || { echo "$(STYLE_ERR)✗ Missing MAVEN_USERNAME$(STYLE_RESET)"; exit 1; }; \
	[ -n "$${MAVEN_PASSWORD:-}" ] || { echo "$(STYLE_ERR)✗ Missing MAVEN_PASSWORD$(STYLE_RESET)"; exit 1; }; \
	printf "$(STYLE_OK)✓ publish credentials present$(STYLE_RESET)\n"

publish-pypi: release-wheel
	$(call banner,Publish: PyPI)
	mkdir -p "$(DIST)"
	$(UV_ENV) $(UV) run --project "$(PY_DIR)" maturin sdist -o "$(abspath $(DIST))"
	@set -eu; \
	if [ "$(PUBLISH_DRY_RUN)" = "1" ]; then \
	  echo "DRY RUN: skipping PyPI upload"; \
	  ls -1 "$(DIST)"; \
	else \
	  [ -n "$${MATURIN_PYPI_TOKEN:-}" ] || { echo "$(STYLE_ERR)✗ Missing MATURIN_PYPI_TOKEN$(STYLE_RESET)"; exit 1; }; \
	  MATURIN_PYPI_TOKEN="$$MATURIN_PYPI_TOKEN" $(UV_ENV) $(UV) run --project "$(PY_DIR)" maturin upload --non-interactive --skip-existing "$(abspath $(DIST))"/*; \
	fi

publish-cargo:
	$(call banner,Publish: crates.io)
	@set -eu; \
	if [ "$(PUBLISH_DRY_RUN)" = "1" ]; then \
	  cargo publish --dry-run --locked --package tdigest-rs; \
	else \
	  [ -n "$${CARGO_REGISTRY_TOKEN:-}" ] || { echo "$(STYLE_ERR)✗ Missing CARGO_REGISTRY_TOKEN$(STYLE_RESET)"; exit 1; }; \
	  CARGO_REGISTRY_TOKEN="$$CARGO_REGISTRY_TOKEN" cargo publish --locked --package tdigest-rs; \
	fi

publish-maven: JAVA_WRAPPER_CHECK
	$(call banner,Publish: Maven)
	@set -eu; \
	if [ "$(PUBLISH_DRY_RUN)" = "1" ]; then \
	  GRADLE_USER_HOME="$(GRADLE_USER_HOME)" "$(GRADLE)" --no-daemon --console=plain -p "$(JAVA_SRC)" publishToMavenLocal; \
	else \
	  [ -n "$${MAVEN_REPOSITORY_URL:-}" ] || { echo "$(STYLE_ERR)✗ Missing MAVEN_REPOSITORY_URL$(STYLE_RESET)"; exit 1; }; \
	  [ -n "$${MAVEN_USERNAME:-}" ] || { echo "$(STYLE_ERR)✗ Missing MAVEN_USERNAME$(STYLE_RESET)"; exit 1; }; \
	  [ -n "$${MAVEN_PASSWORD:-}" ] || { echo "$(STYLE_ERR)✗ Missing MAVEN_PASSWORD$(STYLE_RESET)"; exit 1; }; \
	  MAVEN_REPOSITORY_URL="$$MAVEN_REPOSITORY_URL" MAVEN_USERNAME="$$MAVEN_USERNAME" MAVEN_PASSWORD="$$MAVEN_PASSWORD" MAVEN_SIGNING_KEY="$${MAVEN_SIGNING_KEY:-}" MAVEN_SIGNING_PASSWORD="$${MAVEN_SIGNING_PASSWORD:-}" GRADLE_USER_HOME="$(GRADLE_USER_HOME)" "$(GRADLE)" --no-daemon --console=plain -p "$(JAVA_SRC)" publish; \
	fi

publish: publish-check
	$(MAKE) publish-pypi
	$(MAKE) publish-cargo
	$(MAKE) publish-maven
	@printf "\n$(STYLE_BOLD)==> Publish complete$(STYLE_RESET)\n"
	@printf "$(STYLE_OK)✓ Published to PyPI, crates.io, and Maven (or dry-run equivalent)$(STYLE_RESET)\n"

help:
	@printf "\n$(STYLE_BOLD)Core (dev by default)$(STYLE_RESET)\n"
	@printf "  %-22s %s\n" "setup"          "Install toolchains; sync Python deps (bindings/python) into .venv"
	@printf "  %-22s %s\n" "setup-hooks"    "Install pre-commit hooks with repo-local cache paths"
	@printf "  %-22s %s\n" "build"          "Build Rust lib+CLI (dev), Python ext (dev), Java via Gradle"
	@printf "  %-22s %s\n" "build-rust"     "Build Rust lib+CLI (dev)"
	@printf "  %-22s %s\n" "build-python"   "Build Python extension (maturin develop, dev)"
	@printf "  %-22s %s\n" "build-java"     "Build Java JAR with Gradle (clean+jar)"
	@printf "  %-22s %s\n" "test"           "Run tests: rust + java + python"
	@printf "  %-22s %s\n" "java-test"      "Run Java/JNI tests via Gradle"
	@printf "  %-22s %s\n" "compat-test"    "Run copied upstream Python compatibility tests"
	@printf "  %-22s %s\n" "compat-bench"   "Run copied upstream benchmark manually"
	@printf "  %-22s %s\n" "lint"           "Autofix (ruff/clippy/format) + mypy + rustdoc (deny warnings)"
	@printf "  %-22s %s\n" "clean"          "Remove ALL build artifacts (Rust/Gradle/Python)"
	@printf "\n$(STYLE_BOLD)Releases (release profile)$(STYLE_RESET)\n"
	@printf "  %-22s %s\n" "release-rust"   "Build CLI (release) and smoke"
	@printf "  %-22s %s\n" "release-wheel"  "Build PROD wheel (release) inline and smoke"
	@printf "  %-22s %s\n" "release-jar"    "Gradle jar -> dist/ + latest symlink + smoke"
	@printf "  %-22s %s\n" "release"        "release-rust + release-wheel + release-jar"
	@printf "  %-22s %s\n" "publish-check"  "Validate publish env vars (skipped in dry-run)"
	@printf "  %-22s %s\n" "publish-pypi"   "Build sdist/wheel and upload to PyPI (or skip upload in dry-run)"
	@printf "  %-22s %s\n" "publish-cargo"  "Publish crate to crates.io (or --dry-run)"
	@printf "  %-22s %s\n" "publish-maven"  "Publish Java artifact to Maven repo (or publishToMavenLocal)"
	@printf "  %-22s %s\n" "publish"        "publish-pypi + publish-cargo + publish-maven"
	@printf "\n$(STYLE_BOLD)Publish env vars$(STYLE_RESET)\n"
	@printf "  %-22s %s\n" "PUBLISH_DRY_RUN=1" "Run publish pipeline without uploading to registries"
	@printf "  %-22s %s\n" "MATURIN_PYPI_TOKEN" "PyPI API token for maturin upload"
	@printf "  %-22s %s\n" "CARGO_REGISTRY_TOKEN" "crates.io API token"
	@printf "  %-22s %s\n" "MAVEN_REPOSITORY_URL" "Maven repository publish URL"
	@printf "  %-22s %s\n" "MAVEN_USERNAME/PASSWORD" "Maven repository credentials"
	@printf "  %-22s %s\n" "MAVEN_SIGNING_KEY/PASSWORD" "Optional in-memory PGP signing key/password"
