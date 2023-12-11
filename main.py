from modal import Stub, Image
import torch

from llm_from_scratch.analysis import generate, plot_loss_history
from llm_from_scratch.device import device
from llm_from_scratch.models.multihead_attention import Model
from llm_from_scratch.sample_prep import get_train_validation_split
from llm_from_scratch.tokenizer import get_token_set, encode
from llm_from_scratch.training import train_model


stub = Stub("llm_from_scratch")
image = (
    Image.debian_slim()
    .pip_install_from_pyproject("pyproject.toml")
)


@stub.function(image=image, gpu="any")
def train(text: str):
    vocab = get_token_set(text)
    data = torch.tensor(encode(text, vocab), dtype=torch.long)
    train_data, valid_data = get_train_validation_split(data)

    max_context_length = 256
    print(f"Starting training on {device}")
    model = Model(
        vocab_size=len(vocab),
        max_context_length=max_context_length,
    ).to(device)

    loss_history = train_model(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        max_context_length=max_context_length,
    )
    plot_loss_history(loss_history, "loss_history.png")

    generated_text = generate(
        model=model,
        context=torch.tensor([[vocab.index("S")]]).to(device),
        max_new_tokens=500,
        vocab=vocab,
    )
    print(generated_text)


@stub.local_entrypoint()
def main():
    with open("data/tiny_shakespeare.txt", "r") as f:
        text = f.read()
    train.remote(text)


if __name__ == "__main__":
    with open("data/tiny_shakespeare.txt", "r") as f:
        text = f.read()
    train.local(text)
