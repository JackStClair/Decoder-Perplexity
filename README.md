# Decoder-Perplexity
A Transformer decoder LLM built to calculate perplexity

# Language Model Training with Transformer Architecture

This code implements a language model using a Transformer architecture on the Penn Treebank (PTB) dataset.

The following files are included in the repository:
- `run.py`: The main training script
- `requirements.txt`: A list of required packages
- `README.md`: This README file
- `run.sh`: This can be used to run the code if it is compressed as hw3.zip

## Requirements

The required packages are listed in `requirements.txt`. Key dependencies include:
- PyTorch
- Datasets
- Scikit-learn
- Pandas
- Matplotlib
- Gensim
- tqdm

## Usage

To run the model:

1. Ensure you have all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the training script with your desired submission file name:
   ```bash
   python run.py submission.csv
   ```

The script will:
- Load and preprocess the Penn Treebank dataset
- Initialize a Transformer-based language model
- Train the model using the training set
- Validate performance on the validation set
- Generate perplexitiesfor the test set
- Save results to the specified submission file

The model uses the following hyperparameters:
- Hidden dimension: 512
- Batch size: 32
- Learning rate: 0.0001
- Number of epochs: 50
- Number of attention heads: 16
- Number of transformer layers: 4
- Dropout rate: 0.3
- Embedding dimension: 100

Training progress and metrics will be logged to the console, including:
- Training loss
- Validation loss 
- Average Training Perplexity
- Average Validation Perplexity
