import torch

from llm_from_scratch.device import device


def get_train_validation_split(data: torch.Tensor, valid_pct: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(len(data) * (1 - valid_pct))
    train_data = data[:n]
    valid_data = data[n:]
    return train_data, valid_data


def get_batch(data: torch.Tensor, batch_size: int, max_context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get [batch_size] random contiguous batches from [data], each of length [max_context_length].
    x and y are both of shape (batch_size, max_context_length), and are offset by 1.
    """
    ix = torch.randint(len(data) - max_context_length, (batch_size,))
    return (
        torch.stack([data[i:i+max_context_length] for i in ix]).to(device),
        torch.stack([data[i+1:i+max_context_length+1] for i in ix]).to(device),
    )
