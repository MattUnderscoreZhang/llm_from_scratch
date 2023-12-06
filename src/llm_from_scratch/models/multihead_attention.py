import torch
from torch import nn
from torch.nn import functional as F

from llm_from_scratch.device import device


class Head(nn.Module):
    def __init__(
        self,
        max_context_length: int,  # T
        embedding_size: int,
        head_size: int,
        dropout: float,
    ):
        super().__init__()
        self.head_size = head_size
        self.key_head = nn.Linear(embedding_size, head_size, bias=False)
        self.query_head = nn.Linear(embedding_size, head_size, bias=False)
        self.value_head = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('lower_matrix', torch.tril(torch.ones(max_context_length, max_context_length)))  # lower triangular matrix
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.key_head(x)  # (B, T, head_size)
        queries = self.query_head(x)  # (B, T, head_size)
        values = self.value_head(x)  # (B, T, head_size)
        attention_matrix = queries @ keys.transpose(-2, -1)  # (B, T, T)
        attention_matrix *= self.head_size ** -0.5  # scale by sqrt(head_size)
        T = attention_matrix.shape[-1]
        attention_matrix = attention_matrix.masked_fill(self.lower_matrix[:T, :T] == 0, float("-inf"))  # -inf on top diagonal
        attention_matrix = F.softmax(attention_matrix, dim=-1)  # becomes a prev-value weighter (with weights summing to 1)
        attention_matrix = self.dropout(attention_matrix)  # (B, T, T)
        return attention_matrix @ values  # (B, T, head_size)


class Block(nn.Module):
    def __init__(
        self,
        max_context_length: int,  # T
        embedding_size: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        assert embedding_size % n_heads == 0, "embedding_size must be divisible by n_heads"
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.attention_heads = nn.ModuleList([
            Head(
                max_context_length=max_context_length,
                embedding_size=embedding_size,
                head_size=embedding_size//n_heads,
                dropout=dropout,
            )
            for _ in range(n_heads)
        ])
        self.attention_projection = nn.Linear(embedding_size, embedding_size)
        self.output_layers = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, 4*embedding_size),
            nn.ReLU(),
            nn.Linear(4*embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)  # (B, T, embedding_size)
        weighted_vector = torch.cat(
            [
                attention_head(x)  # (B, T, head_size)
                for attention_head in self.attention_heads
            ],
            dim=-1,
        )  # (B, T, embedding_size)
        x = x + self.attention_projection(weighted_vector)  # (B, T, embedding_size)
        logits = x + self.output_layers(x)  # (B, T, embedding_size)
        return logits


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_context_length: int,  # T
        embedding_size: int = 384,
        n_heads: int = 6,
        n_blocks: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(max_context_length, embedding_size)
        self.max_context_length = max_context_length
        self.blocks = nn.Sequential(
            *[
                Block(
                    max_context_length=max_context_length,
                    embedding_size=embedding_size,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.output_layers = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, vocab_size),
        )

    def _create_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add token and position information to the input tensor.
        """
        token_embedding = self.token_embedding_table(x)  # (B, T, embedding_size)
        position_embedding = self.position_embedding_table(torch.arange(x.shape[1]).to(device))  # (T, embedding_size)
        embedding = token_embedding + position_embedding  # (B, T, embedding_size)
        return embedding

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x and y are each shape (B, T), where each element is an integer in range(vocab_size)
        embedding = self._create_embedding(x)  # (B, T, embedding_size)
        block_output = self.blocks(embedding)  # (B, T, embedding_size)
        logits = self.output_layers(block_output)  # (B, T, vocab_size)
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
