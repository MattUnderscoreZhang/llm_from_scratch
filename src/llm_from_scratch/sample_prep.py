import torch


def get_train_validation_split(data: torch.Tensor, valid_pct: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(len(data) * (1 - valid_pct))
    train_data = data[:n]
    valid_data = data[n:]
    return train_data, valid_data


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get [batch_size] random contiguous batches from [data], each of length [block_size].
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    return (
        torch.stack([data[i:i+block_size] for i in ix]),
        torch.stack([data[i+1:i+block_size+1] for i in ix]),
    )
