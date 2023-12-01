import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import torch
from torch import nn

from llm_from_scratch.models import BigramLanguageModel
from llm_from_scratch.sample_prep import get_train_validation_split
from llm_from_scratch.tokenizer import get_token_set, encode, decode
from llm_from_scratch.training import train_model


def generate(model: nn.Module, context: torch.Tensor, max_new_tokens: int, vocab: list[str]) -> str:
    generated_text = decode(
        model.generate_next_token(
            context,
            max_new_tokens=max_new_tokens,
        ).tolist()[0],
        vocab,
    )
    return generated_text


def plot_loss_history(loss_history: list[torch.Tensor], filename: str) -> None:
    plt.plot(loss_history)
    plt.title("Loss history")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig(filename)


if __name__ == "__main__":
    torch.manual_seed(1337)
    with open("data/tiny_shakespeare.txt", "r") as f:
        text = f.read()

    vocab = get_token_set(text)
    data = torch.tensor(encode(text, vocab), dtype=torch.long)
    train_data, valid_data = get_train_validation_split(data)

    model = BigramLanguageModel(len(vocab))

    """
    generated_text = generate(
        model=model,
        context=torch.tensor([[vocab.index("S")]]),
        max_new_tokens=100,
        vocab=vocab,
    )
    print(generated_text)
    """

    loss_history = train_model(model, train_data)
    plot_loss_history(loss_history, "loss_history.png")
