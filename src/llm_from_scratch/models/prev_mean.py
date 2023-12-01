import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # each token maps onto a next-token distribution
        # thus, here embedding_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x and y are both of shape (batch_size, block_size)
        logits = self.token_embedding_table(x)
        B, T, C = logits.shape
        lower_matrix = torch.tril(torch.ones(T, T))
        avg_matrix = lower_matrix / torch.sum(lower_matrix, dim=1, keepdim=True)
        logits_prev_mean = avg_matrix @ logits
        loss = (
            F.cross_entropy(logits_prev_mean.view(B*T, C), y.reshape(B*T))
            if y is not None
            else torch.Tensor(0)
        )
        return logits, loss

    def generate_next_token(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(x)  # last token in each batch only - (B, C)
            logits = logits[:, -1, :]  # remove dimension - (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            x_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            x = torch.cat((x, x_next), dim=1)  # (B, T+1)
        return x
