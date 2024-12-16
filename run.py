import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from gensim.downloader import load

submission = sys.argv[1]

torch.manual_seed(69)

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Add these lines to check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True) # download the dataset

ptb_train = ptb['train']
ptb_val = ptb['validation']
ptb_test = ptb['test']

train_sentences = []
train_output = []
val_sentences = []
val_output = []
test_sentences = []
test_output = []

for sentence in ptb_train:
    train_sentences.append(["<bos>"] + sentence['sentence'].lower().split() + ["<eos>"])
    train_output.append(sentence['sentence'].lower().split() + ["<eos>", "<pad>"])

for sentence in ptb_val:
    val_sentences.append(["<bos>"] + sentence['sentence'].lower().split() + ["<eos>"])
    val_output.append(sentence['sentence'].lower().split() + ["<eos>", "<pad>"])

for sentence in ptb_test:
    test_sentences.append(["<bos>"] + sentence['sentence'].lower().split() + ["<eos>"])
    test_output.append(sentence['sentence'].lower().split() + ["<eos>", "<pad>"])


class OutputDataset(Dataset):
    """
    Dataset for the output of the encoder and decoder
    """
    def __init__(self, x, y, encoder_vocab=None, decoder_vocab=None, training=True):
        # Create vocabularies if training
        if training:
            self.encoder_vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
            self.decoder_vocab = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}

            # build vocab from training data
            for i in range(len(x)):
                for token in x[i]:
                    if token not in self.encoder_vocab:
                        self.encoder_vocab[token] = len(self.encoder_vocab)
                        self.decoder_vocab[len(self.decoder_vocab)] = token
        else:
            assert encoder_vocab is not None and decoder_vocab is not None
            self.encoder_vocab = encoder_vocab
            self.decoder_vocab = decoder_vocab

        # Convert sentences and indices to integer IDs during initialization
        self.corpus_token_ids = []
        self.corpus_output_ids = []
        for i in range(len(x)):
            token_ids = [self.encoder_vocab.get(token, self.encoder_vocab['<unk>']) for token in x[i]]
            output_ids = [self.encoder_vocab.get(token, self.encoder_vocab['<unk>']) for token in y[i]]
            self.corpus_token_ids.append(torch.tensor(token_ids))
            self.corpus_output_ids.append(torch.tensor(output_ids))

    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        return self.corpus_token_ids[idx], self.corpus_output_ids[idx]

train_dataset = OutputDataset(train_sentences, train_output, training=True)
val_dataset = OutputDataset(val_sentences, val_output, encoder_vocab=train_dataset.encoder_vocab, decoder_vocab=train_dataset.decoder_vocab, training=False)
test_dataset = OutputDataset(test_sentences, test_output, encoder_vocab=train_dataset.encoder_vocab, decoder_vocab=train_dataset.decoder_vocab, training=False)

# collate token_ids and output_ids to make mini-batches
def collate_fn(batch):
    # batch: [(token_ids, output_ids), (token_ids, output_ids), ...]
    
    # Separate token_ids and output_ids
    token_ids = [item[0] for item in batch]
    output_ids = [item[1] for item in batch]
    
    # Pad sequences
    token_ids_padded = pad_sequence(token_ids, batch_first=True, padding_value=train_dataset.encoder_vocab['<pad>'])
    # token_ids_padded.size()  (batch_size, seq_len)
    output_ids_padded = pad_sequence(output_ids, batch_first=True, padding_value=train_dataset.encoder_vocab['<pad>'])
    # output_ids_padded.size()  (batch_size, seq_len)
    return token_ids_padded, output_ids_padded

DIMENSION = 512
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
NUM_HEADS = 16
NUM_LAYERS = 4
DROPOUT = 0.3
EMBEDDING_DIM = 100

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, shuffle=False)

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding for the input tokens
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):  # x: (batch, seq_len, d_model)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, seq_len)
        embedding = self.pos_embedding(pos)  # (1, seq_len, d_model)
        return x + embedding

class TransformerLM(nn.Module):
    """
    Transformer language model
    """
    def __init__(self, vocab_size, embedding_dim, d_model, n_head, n_layer, dropout, pretrained_embeddings=None):
        super().__init__()
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Add projection layer to match dimensions
        self.embedding_projection = nn.Linear(embedding_dim, d_model)
        
        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Rest of the initialization remains the same
        self.pos_encoding = LearnedPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layer)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.causal_mask = torch.triu(torch.ones(1000, 1000) * float('-inf'), diagonal=1)

    def forward(self, x):  # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = self.embedding_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        
        # Generate causal mask with maximum possible sequence length
        max_seq_len = x.size(1)
        causal_mask = self.causal_mask[:max_seq_len, :max_seq_len]  # Ensure mask matches current sequence length
        causal_mask = causal_mask.to(x.device)
        
        out = self.transformer_encoder(x.transpose(0, 1), mask=causal_mask)  # Transpose for transformer_encoder
        out = out.transpose(0, 1)
        
        # This line gets the vocabulary logits
        out = self.output_projection(out)  # (batch, seq_len, vocab_size)
        
        return out

    def generate(self, x, max_len):
        self.eval()
        generated = x
        for _ in range(max_len):
            with torch.no_grad():
                outputs = self(generated)  # (batch_size, seq_len, vocab_size)
                next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)  # Get last token
                generated = torch.cat([generated, next_token], dim=1)  # Append to sequence
        return generated

# Move this function definition before it's used (place it before the model initialization)
def load_glove_embeddings(word_to_idx, embedding_dim=100):
    """
    Load GloVe embeddings from gensim
    """
    print("Loading GloVe embeddings from gensim...")
    
    # Load pre-trained GloVe embeddings from gensim
    glove_vectors = load('glove-wiki-gigaword-100')
    
    # Initialize embedding matrix with random values
    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(word_to_idx), embedding_dim))
    
    # Replace random values with GloVe vectors for words that exist
    found_words = 0
    for word, idx in tqdm(word_to_idx.items(), desc="Processing vocabulary"):
        try:
            if word in glove_vectors:
                embedding_matrix[idx] = glove_vectors[word]
                found_words += 1
        except KeyError:
            continue  # Skip words not in GloVe vocabulary
    
    print(f"Found {found_words}/{len(word_to_idx)} words in GloVe")
    return torch.FloatTensor(embedding_matrix)

# Before model initialization, load GloVe embeddings
pretrained_embeddings = load_glove_embeddings(
    train_dataset.encoder_vocab,
    embedding_dim=EMBEDDING_DIM
)

# Update model initialization
model = TransformerLM(
    vocab_size=len(train_dataset.encoder_vocab),
    embedding_dim=EMBEDDING_DIM,
    d_model=DIMENSION,
    n_head=NUM_HEADS,
    n_layer=NUM_LAYERS,
    dropout=DROPOUT,
    pretrained_embeddings=pretrained_embeddings
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.encoder_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


model = model.to(device)

metrics = []
best_f1 = 0
best_val_loss = float('inf')
best_train_loss = float('inf')
# Training Loop
for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    total_train_loss = 0
    for token_ids, output_ids in train_loader:
        token_ids = token_ids.to(device)
        output_ids = output_ids.to(device)

        optimizer.zero_grad()

        predictions = model(token_ids)  # (batch_size, seq_len, vocab_size)

        targets = output_ids # (batch_size, seq_len)

        # Reshape for loss computation
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1)

        # Compute loss
        loss = loss_fn(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    total_val_loss = 0
    all_predictions = []
    all_output_ids = []

    with torch.no_grad():
        for token_ids, output_ids in val_loader:
            token_ids = token_ids.to(device)
            output_ids = output_ids.to(device)

            outputs = model(token_ids)

            targets = output_ids  # I was testing some shifting of targets, but it was not helpful
            predictions = outputs 

            # Reshape for loss computation
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1)

            # Compute loss
            loss = loss_fn(predictions, targets)
            total_val_loss += loss.item()

            predictions = outputs.argmax(dim=-1)
            mask = output_ids != train_dataset.encoder_vocab['<pad>']

            all_predictions.extend(predictions[mask].tolist())
            all_output_ids.extend(output_ids[mask].tolist())

    # compute train and val loss
    train_loss = total_train_loss / len(train_loader)
    val_loss = total_val_loss / len(val_loader)
    # Calculate perplexity for validation set
    val_perplexity = torch.exp(torch.tensor(val_loss)).item()
    train_perplexity = torch.exp(torch.tensor(train_loss)).item()

    # Calculate metrics
    metrics.append({
    'epoch': epoch + 1,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'train_perplexity': train_perplexity,
    'val_perplexity': val_perplexity
    })
    print(f'epoch = {epoch+1} | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | train_perplexity = {train_perplexity:.3f} | val_perplexity = {val_perplexity:.3f}')

# Get token probabilities for test data
model.eval()
all_batch_perplexity = []
all_sentence_probs = []
total_perplexity = 0
total_tokens = 0

with torch.no_grad():
    for token_ids, output_ids in test_loader:
        token_ids = token_ids.to(device)
        output_ids = output_ids.to(device)

        # Forward pass
        outputs = model(token_ids)  # (batch_size, seq_len, vocab_size)
        
        # Shift logits and output_ids for next-token prediction
        logits = outputs.view(-1, outputs.shape[-1])  # [batch_size * seq_len, vocab_size]
        output_ids_flat = output_ids.view(-1)  # [batch_size * seq_len]

        # Apply softmax to get probabilities
        softmax = F.softmax(logits, dim=-1)
        
        # Get probabilities for the correct tokens
        # Make sure output_ids_flat is properly shaped for gathering
        output_ids_flat = output_ids_flat.unsqueeze(-1)  # Add dimension for gathering
        token_probs = softmax.gather(dim=-1, index=output_ids_flat)
        
        # Take log of probabilities
        log_probs = torch.log(token_probs).squeeze(-1)  # Remove gathering dimension
        
        # Create mask for valid tokens
        mask = output_ids_flat.squeeze(-1) != train_dataset.encoder_vocab['<pad>']
        
        # Calculate perplexity for valid tokens
        valid_log_probs = log_probs[mask]
        batch_perplexity = torch.exp(-valid_log_probs.mean()).item()  # Negative because we want negative log likelihood
        
        all_batch_perplexity.append(batch_perplexity)

# Final perplexity
for_csv = pd.DataFrame({'ID': range(len(all_batch_perplexity)), 'ppl': all_batch_perplexity})
for_csv.to_csv(submission, index=False)
"""
# After training loop
train_losses = [m['train_loss'] for m in metrics]
val_losses = [m['val_loss'] for m in metrics]
train_perplexities = [m['train_perplexity'] for m in metrics]
val_perplexities = [m['val_perplexity'] for m in metrics]
epochs = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, val_losses, label='Val Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_perplexities, label='Train Perplexity', color='red')
plt.plot(epochs, val_perplexities, label='Val Perplexity', color='green')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()
plt.savefig('perplexity_plot.png')
plt.close()
"""