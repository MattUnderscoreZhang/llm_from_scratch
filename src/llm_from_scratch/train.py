import torch
from torch import nn
from torch.nn import functional as F


def get_train_validation_split(data: torch.Tensor, valid_pct: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(len(data) * (1 - valid_pct))
    train_data = data[:n]
    valid_data = data[n:]
    return train_data, valid_data


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> torch.Tensor:
    """
    Get [batch_size] random contiguous batches from [data], each of length [block_size].
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    return torch.stack([data[i:i+block_size+1] for i in ix])


def train_with_batch(batch: torch.Tensor) -> None:
    batch_size = batch.shape[0]
    block_size = batch.shape[1] - 1
    for b in range(batch_size):
        for t in range(block_size):
            context = batch[b, :t+1]
            target = batch[b, t+1]
            return context, target


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # each token maps onto a next-token distribution
        # thus, here embedding_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x and y are both of shape (batch_size, block_size)
        # output is of shape (batch_size, block_size, embedding_size)
        logits = self.token_embedding_table(x)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        y = y.reshape(B*T)
        loss = F.cross_entropy(logits, y)
        return logits, loss
