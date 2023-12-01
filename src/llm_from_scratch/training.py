import torch

from llm_from_scratch.sample_prep import get_batch


def train_model(model: torch.nn.Module, train_data: torch.Tensor) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 32
    n_batches = 100

    loss = torch.Tensor(0.0)
    for _ in range(n_batches):
        x, y = get_batch(train_data, batch_size, block_size=8)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return loss.item()


def train_with_batch(x: torch.Tensor, y: torch.Tensor) -> None:
    batch_size, block_size = x.shape
    for b in range(batch_size):
        for t in range(block_size):
            context = x[b, :t+1]
            target = y[b, t]
            print(context, target)
