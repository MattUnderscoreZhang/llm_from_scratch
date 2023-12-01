import torch
from torch import nn

from llm_from_scratch.models import BigramLanguageModel
from llm_from_scratch.sample_prep import get_train_validation_split, get_batch
from llm_from_scratch.tokenizer import get_token_set, encode, decode


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

    model = BigramLanguageModel(len(vocab))

    x, y = get_batch(train_data, batch_size=4, block_size=8)
    # train_with_batch(x, y)
    # logits, loss = model(x, y)

    generated_text = generate(
        model=model,
        context=torch.tensor([[vocab.index("S")]]),
        max_new_tokens=100,
        vocab=vocab,
    )
    print(generated_text)
