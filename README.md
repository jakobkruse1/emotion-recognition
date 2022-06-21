# Comparing unimodal and multimodal emotion classification systems on cohesive data
*Master's Thesis by Jakob Kruse*

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/downloads/release/python-380/)
[![codecov](https://codecov.io/gh/jakobkruse1/emotion-recognition/branch/main/graph/badge.svg?token=2PUCAJG0XA)](https://codecov.io/gh/jakobkruse1/emotion-recognition)
[![GitHub license](https://badgen.net/github/license/jakobkruse1/emotion-recognition)](https://github.com/jakobkruse1/emotion-recognition/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/-Docs-green)](https://jakobkruse1.github.io/emotion-recognition)
[![Linkedin](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/jakob-kruse-b7293a197/)

This is the repository for my master's thesis on emotion detection.
This is still a WIP.


### Setup
I recommend setting up a new virtual environment in the venv folder.
Then, install the required packages:
```bash
pip install -r requirements.txt
```
To download the datasets, please refer to [data/DESCRIPTION.md](data/DESCRIPTION.md).
Some of the datasets are publicly available, to download them use:
```bash
bash data/prepare_all.sh
```
Prepare all the classification models:
```bash
bash models/install_all.sh
```

### Tests
Run the tests using:
```bash
python -m pytest tests
```
Run the tests with coverage:
```bash
python -m pytest --cov=src/ tests/
```

### Extras
Install the pre-commit hooks for linting:
```python
pre-commit install
```
To run all linters manually, use:
```python
pre-commit
```
Note: only changes added to git are included in linting when using the pre-commit command.

You can also run the single linters one at a time:
```bash
black --check .
flake8 .
isort --check-only .
```
