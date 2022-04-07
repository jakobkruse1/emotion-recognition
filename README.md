# Comparing unimodal and multimodal emotion classification systems on cohesive data
*Master's Thesis by Jakob Kruse*

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/downloads/release/python-380/)
[![codecov](https://codecov.io/gh/jakobkruse1/thesis-emotion-detection/branch/main/graph/badge.svg?token=2PUCAJG0XA)](https://codecov.io/gh/jakobkruse1/thesis-emotion-detection)
[![GitHub license](https://badgen.net/github/license/jakobkruse1/thesis-emotion-detection)](https://github.com/jakobkruse1/thesis-emotion-detection/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/-Docs-green)](https://jakobkruse1.github.io/thesis-emotion-detection)
[![Linkedin](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/jakob-kruse-b7293a197/)

This is the repository for my master's thesis on emotion detection.
This is still a WIP.


### Setup
I recommend setting up a new virtual environment in the venv folder.
Then, install the required packages:
```bash
pip install -r requirements.txt
```
Download all the datasets:
```bash
bash data/prepare_all.sh
```
Install all the classification models:
```bash
bash models/install_all.sh
```

### Tests
Run the tests using:
```python
python -m pytest tests
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
