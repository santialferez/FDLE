"""Tokenization, vocabulary building and text batches."""

import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset


def tokenize(text: str) -> list[str]:
    """Return lowercase letters and digit tokens, keeping common apostrophes."""
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.lower().replace("’", "'"))


class SimpleVocab:
    """Map frequent training tokens to integer IDs for an embedding layer."""

    pad_idx = 0
    unk_idx = 1

    def __init__(self, max_size=10000, min_freq=2):
        if max_size < 2:
            raise ValueError("The vocabulary needs room for padding and unknown tokens.")
        self.max_size = max_size
        self.min_freq = min_freq
        self.token_to_idx = {}
        self._idx_to_token = ["<pad>", "<unk>"]

    def fit(self, texts):
        """Build the vocabulary from training texts and return it."""
        counts = Counter(token for text in texts for token in tokenize(text))
        words = [word for word, count in counts.most_common(self.max_size - 2)
                 if count >= self.min_freq]
        self.token_to_idx = {word: idx for idx, word in enumerate(words, start=2)}
        self._idx_to_token = ["<pad>", "<unk>", *words]
        return self

    def encode(self, text, max_len=150) -> list[int]:
        """Convert a text to IDs and truncate or right-pad to max_len."""
        if max_len < 0:
            raise ValueError("max_len must be nonnegative.")
        ids = [self.token_to_idx.get(token, self.unk_idx)
               for token in tokenize(text)[:max_len]]
        return ids + [self.pad_idx] * (max_len - len(ids))

    def decode(self, ids, skip_special=True) -> str:
        """Join decoded tokens with spaces, optionally keeping <pad> and <unk>."""
        tokens = []
        for value in ids:
            idx = int(value)
            if not 0 <= idx < len(self._idx_to_token):
                idx = self.unk_idx
            if skip_special and idx in (self.pad_idx, self.unk_idx):
                continue
            tokens.append(self._idx_to_token[idx])
        return " ".join(tokens)

    def __len__(self):
        return len(self.token_to_idx) + 2


def make_text_loaders(train_texts, train_y, val_texts, val_y, vocab, *,
                      max_len=150, batch_size=64) -> tuple[DataLoader, DataLoader]:
    """Encode both splits with the given vocabulary and shuffle training rows."""
    def make_loader(texts, labels, shuffle):
        encoded = [vocab.encode(text, max_len) for text in texts]
        inputs = torch.tensor(encoded, dtype=torch.long).reshape(len(encoded), max_len)
        targets = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
        return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=shuffle)

    return make_loader(train_texts, train_y, True), make_loader(val_texts, val_y, False)
