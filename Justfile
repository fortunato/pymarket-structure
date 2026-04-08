# Justfile — task runner for market-structure.
# Run `just` (no args) to list available recipes.
# Docs: https://just.systems/man/en/

set shell := ["bash", "-uc"]

# Default recipe — show the list when you run `just` with no args.
default:
    @just --list --unsorted

# ─── Code quality ──────────────────────────────────────────────────────────

# Run the full CI pipeline: lint → format-check → type → test.
check: lint-check format-check type test

# Lint and auto-fix.
lint:
    uv run ruff check --fix .

# Lint without writing fixes (CI mode).
lint-check:
    uv run ruff check .

# Format the codebase.
fmt:
    uv run ruff format .

# Verify formatting without writing.
format-check:
    uv run ruff format --check .

# Strict type check across src/ and tests/.
type:
    uv run pyright

# ─── Tests ─────────────────────────────────────────────────────────────────

# Examples:
#   just test                           # all tests
#   just test -k hello                  # filter by name
#   just test tests/test_package.py     # specific file
#   just test -x --pdb                  # stop on first failure, drop into pdb
[doc("Run the test suite with coverage. Pass extra pytest args through.")]
test *args:
    uv run pytest {{args}}

# Generate an HTML coverage report and print its path.
cov-html:
    uv run pytest --cov-report=html
    @echo "Open: file://$(pwd)/htmlcov/index.html"

# ─── Pre-commit ────────────────────────────────────────────────────────────

# Run all pre-commit hooks against every file in the repo.
hooks:
    uv run pre-commit run --all-files

# Update pre-commit hook versions to their latest releases.
hooks-update:
    uv run pre-commit autoupdate

# ─── Dependencies ──────────────────────────────────────────────────────────

# Sync the venv from uv.lock (run after pulling, after editing pyproject.toml).
sync:
    uv sync

# Refresh the lockfile to the latest compatible versions.
lock-upgrade:
    uv lock --upgrade
