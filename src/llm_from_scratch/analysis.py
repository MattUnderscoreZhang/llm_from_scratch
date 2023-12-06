import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import torch
from torch import nn

from llm_from_scratch.tokenizer import decode


def generate(model: nn.Module, context: torch.Tensor, max_new_tokens: int, vocab: list[str]) -> str:
    generated_text = decode(
        model.generate_next_tokens(
            context,
            max_new_tokens=max_new_tokens,
        ).tolist()[0],
        vocab,
    )
    return generated_text


def plot_loss_history(loss_history: dict[str, list[float]], filename: str) -> None:
    plt.plot(loss_history["training"], color="orange")
    plt.plot(loss_history["validation"], color="blue")
    plt.title("Loss history")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.savefig(filename)
