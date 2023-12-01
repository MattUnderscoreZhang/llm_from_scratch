import torch

from llm_from_scratch.tokenizer import get_token_set, encode, decode
from llm_from_scratch.train import (
    get_train_validation_split,
    get_batch,
    train_with_batch,
    BigramLanguageModel,
)


if __name__ == "__main__":
    torch.manual_seed(1337)
    with open("data/tiny_shakespeare.txt", "r") as f:
        text = f.read()

    vocab = get_token_set(text)
    data = torch.tensor(encode(text, vocab), dtype=torch.long)
    train_data, valid_data = get_train_validation_split(data)

    model = BigramLanguageModel(len(vocab))

    batch = get_batch(train_data, batch_size=4, block_size=8)
    # train_with_batch(batch)
    out = model(batch[:, :-1], batch[:, 1:])
    print(out.shape)
