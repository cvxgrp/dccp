# cvx-package-template

[![build](https://github.com/langestefan/cvx-package-template/actions/workflows/release.yaml/badge.svg)](https://github.com/langestefan/cvx-package-template/actions/workflows/build.yml)
[![docs](https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat)](https://langestefan.github.io/cvx-package-template/)
[![codecov](https://codecov.io/gh/langestefan/cvx-package-template/graph/badge.svg?token=WQKQEUOS8B)](https://codecov.io/gh/langestefan/cvx-package-template)
[![license](https://img.shields.io/github/license/langestefan/cvx-package-template)](https://github.com/langestefan/cvx-package-template/blob/main/LICENSE)
[![pypi](https://img.shields.io/pypi/v/cvx-package-template)](https://pypi.org/project/cvx-package-template/)

## Template instructions (to be removed)

This is a template for creating packages in the cvxpy ecosystem. It provides a basic
structure and configuration for a Python package, including:

- A `pyproject.toml` file for package metadata and dependencies.
- A `tests` directory for unit tests using `pytest` and `pytest-cov` for coverage
reporting.
- A `docs` directory for documentation using Sphinx.
- A `examples` directory for example usage of the package, which will be displayed in
the documentation.
- Linting and formatting using `ruff`.
- Pre-commit hooks using `pre-commit` to ensure code quality before committing changes.

## Running tests

To be able to run unit tests with [uv](https://github.com/astral-sh/uv) you will need:

```bash
uv sync --group dev
```

You can then run the tests using:

```bash
uv run pytest tests
```

Alternatively, with `pip` you can install the `dev` dependencies and run the tests using:

```bash
pip install -e .[dev]
pytest tests
```

## Building documentation locally

To build and run the documentation locally using `sphinx-autobuild`,
you need to first install dependencies using the following commands:

```bash
uv sync --group dev --group doc
uv run sphinx-autobuild docs/src docs/_build/html
```

Alternatively, with `pip` you can install the `dev` and `doc` dependencies and run the documentation using:

```bash
pip install -e .[dev,doc]
sphinx-autobuild docs/src docs/_build/html
```

## Repository description goes here.

The full documentation is available [here](https://www.cvxgrp.org/repository/).

If you wish to cite repository please cite the papers listed [here](https://www.cvxgrp.org/repository/citing).
