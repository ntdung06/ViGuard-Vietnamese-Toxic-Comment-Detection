from collections import Counter
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def vocabulary(texts, max_size=30000):
    counter = Counter(token for text in texts for token in text.split())
    vocab = {'<pad>': 0, '<unk>': 1}
    for token, _ in counter.most_common(max_size - 2):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_batch(texts, vocab, max_length=128):
    rows = [[vocab.get(token, 1) for token in text.split()[:max_length]] or [1] for text in texts]
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
    tokens = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long)
    for i, row in enumerate(rows):
        tokens[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    return tokens, lengths


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 3)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=1)))
