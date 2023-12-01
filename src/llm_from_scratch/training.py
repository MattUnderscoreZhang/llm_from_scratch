import torch
from torch import nn

from llm_from_scratch.sample_prep import get_batch


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    valid_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int = 50,
) -> dict[str, float]:
    avg_loss = {}
    model.eval()
    for data, label in zip([train_data, valid_data], ["training", "validation"]):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size)
            _, loss = model(x, y)
            losses[i] = loss.item()
        avg_loss[label] = losses.mean().item()
    model.train()
    return avg_loss


def train_model(
    model: nn.Module,
    train_data: torch.Tensor,
    valid_data: torch.Tensor,
    batch_size: int = 32,
    block_size: int = 8,
    n_batches: int = 10_000,
    calculate_loss_every: int = 100,
) -> dict[str, list[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_history = {"training": [], "validation": []}
    for i in range(n_batches):
        x, y = get_batch(train_data, batch_size, block_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i % calculate_loss_every == 0:
            losses = estimate_loss(model, train_data, valid_data, batch_size, block_size)
            loss_history["training"].append(losses["training"])
            loss_history["validation"].append(losses["validation"])
    return loss_history
