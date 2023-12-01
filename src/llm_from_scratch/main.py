import torch
from torch import nn

from llm_from_scratch.analysis import plot_loss_history
from llm_from_scratch.device import device
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


if __name__ == "__main__":
    torch.manual_seed(1337)
    with open("data/tiny_shakespeare.txt", "r") as f:
        text = f.read()

    vocab = get_token_set(text)
    data = torch.tensor(encode(text, vocab), dtype=torch.long)
    train_data, valid_data = get_train_validation_split(data)

    model = BigramLanguageModel(len(vocab)).to(device)

    loss_history = train_model(model, train_data, valid_data)
    plot_loss_history(loss_history, "loss_history.png")

    generated_text = generate(
        model=model,
        context=torch.tensor([[vocab.index("S")]]).to(device),
        max_new_tokens=500,
        vocab=vocab,
    )
    print(generated_text)
