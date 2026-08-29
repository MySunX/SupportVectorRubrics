from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if logits.numel() == 0:
        return logits
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(shifted, dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim=dim) - 1
    view_shape = [1] * shifted.dim()
    view_shape[dim] = shifted.size(dim)
    ks = torch.arange(
        1,
        shifted.size(dim) + 1,
        device=shifted.device,
        dtype=shifted.dtype,
    ).view(view_shape)
    support = z_sorted > (z_cumsum / ks)
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = z_cumsum.gather(dim, support_size - 1) / support_size.to(shifted.dtype)
    return torch.clamp(shifted - tau, min=0.0)


@dataclass
class VectorizerConfig:
    max_features: int = 4096
    min_df: int = 1
    ngram_range: tuple[int, int] = (1, 2)


class PromptVectorizer:
    def __init__(self, config: VectorizerConfig | None = None):
        self.config = config or VectorizerConfig()
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            ngram_range=self.config.ngram_range,
            strip_accents="unicode",
        )

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)

    @property
    def input_dim(self) -> int:
        return len(self.vectorizer.get_feature_names_out())

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, path: str) -> "PromptVectorizer":
        with open(path, "rb") as f:
            vectorizer = pickle.load(f)
        obj = cls()
        obj.vectorizer = vectorizer
        return obj


class PromptSelectorModel(torch.nn.Module):
    def __init__(self, *, input_dim: int, bank_size: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = int(input_dim)
        self.bank_size = int(bank_size)
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim > 0:
            self.hidden = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.output = torch.nn.Linear(self.hidden_dim, self.bank_size)
        else:
            self.hidden = None
            self.output = torch.nn.Linear(self.input_dim, self.bank_size)
        self.raw_global_weights = torch.nn.Parameter(torch.zeros(self.bank_size))

    def forward(self, prompt_features: torch.Tensor) -> torch.Tensor:
        if self.hidden is not None:
            hidden = torch.relu(self.hidden(prompt_features))
            logits = self.output(hidden)
        else:
            logits = self.output(prompt_features)
        return sparsemax(logits, dim=-1)

    def global_weights(self) -> torch.Tensor:
        return F.softplus(self.raw_global_weights)

    def score_bank(self, prompt_features: torch.Tensor) -> torch.Tensor:
        alpha = self(prompt_features)
        return alpha * self.global_weights().unsqueeze(0)

    def prune_output(self, keep_indices: list[int]) -> None:
        keep_tensor = torch.as_tensor(keep_indices, dtype=torch.long)
        device = self.raw_global_weights.device
        keep_tensor = keep_tensor.to(device)

        old_output = self.output
        new_output = torch.nn.Linear(old_output.in_features, len(keep_indices))
        new_output.weight.data.copy_(old_output.weight.data[keep_tensor].cpu())
        new_output.bias.data.copy_(old_output.bias.data[keep_tensor].cpu())
        self.output = new_output.to(device)

        new_raw_weights = torch.nn.Parameter(self.raw_global_weights.data[keep_tensor].clone())
        self.raw_global_weights = new_raw_weights
        self.bank_size = len(keep_indices)
