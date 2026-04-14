# Publishing & Release Guide

This guide covers the one-time setup for automated PyPI publishing and the ongoing release process.

## One-Time Setup: Trusted Publishers (OIDC)

Trusted Publishers let GitHub Actions publish to PyPI without long-lived API tokens. OIDC tokens are short-lived and cryptographically bound to the workflow.

### Step 1: Create GitHub Environments

GitHub environments are required for the OIDC trust relationship.

1. Go to your repository on GitHub → **Settings** → **Environments**
2. Create an environment named **`pypi`**
   - Optionally add deployment protection rules (e.g., require manual approval)
3. Create an environment named **`testpypi`**
   - No protection rules needed (test environment)

### Step 2: Configure Trusted Publisher on PyPI

Since the `market-structure` package hasn't been published yet, set up a "pending publisher" — this reserves the package name and pre-authorizes your workflow.

#### Production PyPI

1. Log in to https://pypi.org
2. Go to **Account Settings** → **Publishing** → **Add a new pending publisher**
3. Fill in:
   - **PyPI project name**: `market-structure`
   - **Owner**: your GitHub username or org (must match exactly)
   - **Repository name**: `py-market-structure-dev`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`
4. Click **Add**

#### TestPyPI

1. Log in to https://test.pypi.org (separate account from PyPI)
2. Go to **Account Settings** → **Publishing** → **Add a new pending publisher**
3. Fill in:
   - **PyPI project name**: `market-structure`
   - **Owner**: same as above
   - **Repository name**: `py-market-structure-dev`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `testpypi`
4. Click **Add**

### Fallback: API Token

If Trusted Publishers can't be used (e.g., organization restrictions):

1. On PyPI → **Account Settings** → **API tokens** → **Add API token**
2. Scope: project-level (select `market-structure`)
3. Copy the token (starts with `pypi-`)
4. On GitHub → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**
   - Name: `PYPI_API_TOKEN`
   - Value: paste the token
5. Modify `publish.yml` to use:
   ```yaml
   - uses: pypa/gh-action-pypi-publish@release/v1
     with:
       password: ${{ secrets.PYPI_API_TOKEN }}
   ```

Repeat for TestPyPI with secret name `TEST_PYPI_API_TOKEN` and `repository-url: https://test.pypi.org/legacy/`.

---

## Semantic Versioning

This project follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html):

- **MAJOR** (X.0.0): Breaking changes to the public API
- **MINOR** (0.X.0): New features, backwards-compatible
- **PATCH** (0.0.X): Bug fixes, backwards-compatible

The version is maintained in a single source of truth: the `version` field in `pyproject.toml`.

### Pre-release Versions

- Release candidates: `0.2.0rc1`, `0.2.0rc2`, ...
- RC tags publish to TestPyPI (not production PyPI)

---

## Changelog

The changelog follows the [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) format.

### How to Update

1. As you work, add entries under the `[Unreleased]` section in `CHANGELOG.md`
2. Group entries by category: **Added**, **Changed**, **Fixed**, **Removed**
3. At release time, rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD` and add a fresh `[Unreleased]` section

### What Warrants a Changelog Entry

- New public API functions or classes → **Added**
- Changed behavior of existing functions → **Changed**
- Bug fixes → **Fixed**
- Removed or deprecated functions → **Removed**

Internal refactors and CI changes generally don't need changelog entries.

---

## Release Process

### 1. Prepare the Release

```bash
# Update version in pyproject.toml
# e.g., change version = "0.1.0" to version = "0.2.0"

# Update CHANGELOG.md
# Move items from [Unreleased] to [0.2.0] - 2026-XX-XX

# Commit
git add pyproject.toml CHANGELOG.md
git commit -m "release: v0.2.0"
git push
```

### 2. Create the GitHub Release

1. Go to your repository → **Releases** → **Create a new release**
2. **Tag**: `v0.2.0` (must match the version in `pyproject.toml`)
3. **Title**: `v0.2.0`
4. **Description**: Copy the relevant section from CHANGELOG.md
5. Click **Publish release**

### 3. What Happens Automatically

The `publish.yml` workflow:
1. Runs the full test suite
2. Validates that the tag version matches `pyproject.toml`
3. Builds the sdist and wheel with `uv build`
4. Publishes to PyPI via Trusted Publishers (OIDC)

### 4. Verify

```bash
pip install market-structure==0.2.0
```

### Testing with a Release Candidate

To test the publish pipeline without affecting production PyPI:

1. Set version to `0.2.0rc1` in `pyproject.toml`
2. Commit and push
3. Create a GitHub release with tag `v0.2.0rc1`, check "This is a pre-release"
4. The workflow publishes to TestPyPI instead

```bash
pip install --index-url https://test.pypi.org/simple/ market-structure==0.2.0rc1
```

---

## Release Checklist

- [ ] Version bumped in `pyproject.toml`
- [ ] `CHANGELOG.md` updated — items moved from `[Unreleased]` to new version
- [ ] Committed and pushed to main
- [ ] GitHub release created with matching `vX.Y.Z` tag
- [ ] Publish workflow succeeded
- [ ] Package installable: `pip install market-structure==X.Y.Z`
