import torch
from torch import nn
from torch.nn import functional as F

from llm_from_scratch.device import device


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_context_length: int,  # T
        embedding_size: int = 32,
        head_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(max_context_length, embedding_size)
        self.max_context_length = max_context_length
        self.head_size = head_size
        self.key_head = nn.Linear(embedding_size, head_size, bias=False)
        self.query_head = nn.Linear(embedding_size, head_size, bias=False)
        self.value_head = nn.Linear(embedding_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc_layer = nn.Linear(head_size, vocab_size)

    def _create_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add token and position information to the input tensor.
        """
        token_embedding = self.token_embedding_table(x)  # (B, T, embedding_size)
        position_embedding = self.position_embedding_table(torch.arange(x.shape[1]).to(device))  # (T, embedding_size)
        embedding = token_embedding + position_embedding  # (B, T, embedding_size)
        return embedding

    def _apply_attention_head(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.key_head(x)  # (B, T, head_size)
        queries = self.query_head(x)  # (B, T, head_size)
        values = self.value_head(x)  # (B, T, head_size)
        attention_matrix = queries @ keys.transpose(-2, -1)  # (B, T, T)
        attention_matrix *= self.head_size ** -0.5  # scale by sqrt(head_size)
        T = attention_matrix.shape[-1]
        lower_matrix = torch.tril(torch.ones(T, T))  # lower triangular matrix
        attention_matrix = attention_matrix.masked_fill(lower_matrix == 0, float("-inf"))  # -inf on top diagonal
        attention_matrix = F.softmax(attention_matrix, dim=-1)  # becomes a prev-value weighter (with weights summing to 1)
        attention_matrix = self.dropout(attention_matrix)  # (B, T, T)
        return attention_matrix @ values  # (B, T, head_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x and y are each shape (B, T), where each element is an integer in range(vocab_size)
        embedding = self._create_embedding(x)  # (B, T, embedding_size)
        weighted_vectors = self._apply_attention_head(embedding)  # (B, T, head_size)
        logits = self.fc_layer(weighted_vectors)  # (B, T, vocab_size)
        B, T, C = logits.shape
        loss = (
            F.cross_entropy(logits.view(B*T, C), y.reshape(B*T))
            if y is not None
            else torch.Tensor(0)
        )
        return logits, loss

    def generate_next_tokens(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(x[:, -self.max_context_length:])  # truncate input vectors
            logits = logits[:, -1, :]  # remove dimension - (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            x_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            x = torch.cat((x, x_next), dim=1)  # (B, T+1)
        return x
