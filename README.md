# Comparing unimodal and multimodal emotion classification systems on cohesive data
*Master's Thesis by Jakob Kruse*

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/downloads/release/python-380/)
[![codecov](https://codecov.io/gh/jakobkruse1/emotion-recognition/branch/main/graph/badge.svg?token=2PUCAJG0XA)](https://codecov.io/gh/jakobkruse1/emotion-recognition)
[![GitHub license](https://badgen.net/github/license/jakobkruse1/emotion-recognition)](https://github.com/jakobkruse1/emotion-recognition/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/-Docs-green)](https://jakobkruse1.github.io/emotion-recognition)
[![Linkedin](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/jakob-kruse-b7293a197/)

This repository serves as a framework for emotion classification. It implements several classifiers
for different data types/modalities like face images, text/sentences, voice/speech, physiological signals (heartrate, acceleration from a smartwatch) and more sensors.
The classifiers can be trained and then used for inference. For training, recommended datasets and how to set them up
is described in [data/DESCRIPTION.md](data/DESCRIPTION.md).

**Emotions**: All classifiers used in this work use 7 emotion classes. The classes are the six basic emotions defined by Paul Ekman:
*happiness, sadness, anger, fear, surprise, disgust*; and an additional *neutral* class. Existing emotion classification models have been retrained on these seven classes
in order to make the models comparable and have a single interface for all models.

**Classifiers**: All classifiers use the same interface for emotion classification.


## ⚙️ Setup
I recommend setting up a new virtual environment in the venv folder.
<details>
<summary>How to set up a virtual environment in Python 3</summary>

```
sudo apt install python3-pip python3-venv
python -m venv venv
source venv/bin/activate
```
</details>

Then, install the required packages:
```bash
pip install -r requirements.txt
```
To download the datasets, please refer to [data/DESCRIPTION.md](data/DESCRIPTION.md).
Some of the datasets are publicly available, to download them you can use:
```bash
bash data/prepare_all.sh
```
Prepare all the classification models:
```bash
bash models/install_all.sh
```

## ⬇️ Download models
To download the pretrained models described in my thesis, use the script:
```bash
bash models/download_model.sh --data <modality> --model <model_name>
```
Currently, only the best model for each modality is available. For details on
the available options for `--data` and `--model`, please refer to the help:
```bash
bash models/download_model.sh -h
```

## 🧪 Testing
Run the tests using:
```bash
python -m pytest tests
```
Run the tests with coverage:
```bash
python -m pytest --cov=src/ tests/
```

## 🔮 Overview of the files

|                              |                                                                        |
|------------------------------|------------------------------------------------------------------------|
| 📂 `.github`                 | Folder containing Github CI configurations.                            |
| 📂 `data`                    | Folder that contains all the training and evaluation datasets.         |
| 📂 `docs`                    | Folder that contains the documentation wiht sphinx.                    |
| 📂 `experiments`             | Folder with experiment configurations and scripts for training models. |
| 📂 `models`                  | Folder that contains trained models for emotion classification.        |
| 📂 `src`                     | Folder with source code for this project.                              |
| 📃 `.gitignore`              | Files that are ignored by git.                                         |
| 📃 `.pre-commit-config.yaml` | Linter configuration file.                                             |
| 📃 `LICENSE`                 | MIT License file.                                                      |
| 📃 `pyproject.toml`          | Linter and Test configurations.                                        |
| 📃 `README.md`               | Explanation (You are here).                                            |
| 📃 `requirements.txt`        | Python Package requirements for the project.                           |
| 📃 `setup.cfg`               | Another linter configuration file.                                     |

## ✒️ Linters
Install the pre-commit hooks for linting:
```python
pre-commit install
```
To run all linters manually, use:
```python
pre-commit
```
Note: only changes added to git are included in linting when using the pre-commit command.

You can also run the single linters one at a time to apply the linter to unstaged files:
```bash
black --check .
flake8 .
isort --check-only .
```
