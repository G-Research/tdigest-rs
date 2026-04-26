# Publishing Guide

This project supports publishing to PyPI, crates.io, and Maven Central.

## GitHub Actions release workflows
Release workflows are in `.github/workflows/`:
- `release_pypi.yml`
- `release_cargo.yml`
- `release_maven.yml`

They trigger on tags matching `v*`.

### 1. Configure GitHub environments and secrets

PyPI (`pypi` environment):
- Configure PyPI Trusted Publisher for this repo/workflow.

Cargo (`crates-io` environment):
- Secret: `CARGO_REGISTRY_TOKEN`

Maven (`maven` environment):
- Secrets: `MAVEN_REPOSITORY_URL`, `MAVEN_USERNAME`, `MAVEN_PASSWORD`
- If signing is required: `MAVEN_SIGNING_KEY`, `MAVEN_SIGNING_PASSWORD`

### 2. Create and push a release tag
- Ensure `Cargo.toml` version matches the tag without `v`.
- Example:

```bash
git tag v0.2.4
git push origin v0.2.4
```

## Local publish
Use `make publish` when publishing from local credentials.

Dry run:

```bash
PUBLISH_DRY_RUN=1 make publish
```

Real publish:

```bash
MATURIN_PYPI_TOKEN=... \
CARGO_REGISTRY_TOKEN=... \
MAVEN_REPOSITORY_URL=... \
MAVEN_USERNAME=... \
MAVEN_PASSWORD=... \
make publish
```

Optional signing variables:
- `MAVEN_SIGNING_KEY`
- `MAVEN_SIGNING_PASSWORD`

## Repository rulesets
Recommended repository protection and ruleset setup is documented in `.github/REPO_SETTINGS.md`.
