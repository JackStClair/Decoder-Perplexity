# Decoder-Perplexity

Transformer decoder language model for next-token prediction and perplexity scoring on the Penn Treebank (PTB) dataset.

![Build Status](https://img.shields.io/badge/build-not_configured-lightgrey)
![Tests](https://img.shields.io/badge/tests-not_configured-lightgrey)
![Coverage](https://img.shields.io/badge/coverage-not_configured-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

## Project Overview

This project:
- loads and preprocesses PTB text data
- builds a decoder-only Transformer language model
- trains and validates the model
- computes perplexity on test samples
- writes predictions to a CSV submission file

## Repository Contents

- `run.py` - training, validation, and test perplexity pipeline
- `requirements.txt` - Python dependencies
- `run.sh` - helper shell script for running the project
- `README.md` - project documentation

## Requirements

- Python `3.10+` recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run training and generate a submission file:

```bash
python run.py submission.csv
```

Output:
- `submission.csv` with `ID` and `ppl` columns

## Model Configuration

Current defaults in `run.py`:

- hidden dimension (`DIMENSION`): `512`
- batch size (`BATCH_SIZE`): `32`
- learning rate (`LEARNING_RATE`): `0.0001`
- epochs (`NUM_EPOCHS`): `50`
- attention heads (`NUM_HEADS`): `16`
- Transformer layers (`NUM_LAYERS`): `4`
- dropout (`DROPOUT`): `0.3`
- embedding dimension (`EMBEDDING_DIM`): `100`

## Training Logs

During training, the script logs:
- training loss
- validation loss
- training perplexity
- validation perplexity
