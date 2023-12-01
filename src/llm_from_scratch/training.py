import torch


def train_model(model: torch.nn.Module) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


def train_with_batch(x: torch.Tensor, y: torch.Tensor) -> None:
    batch_size, block_size = x.shape
    for b in range(batch_size):
        for t in range(block_size):
            context = x[b, :t+1]
            target = y[b, t]
            print(context, target)
