import torch


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


"""
def train_with_batch(batch: torch.Tensor) -> None:
    batch_size = batch.shape[0]
    block_size = batch.shape[1] - 1
    for b in range(batch_size):
        for t in range(block_size):
            context = batch[b, :t+1]
            target = batch[b, t+1]
"""