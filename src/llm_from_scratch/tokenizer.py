def get_token_set(text: str) -> list[str]:
    chars = sorted(list(set(text)))
    return chars


def encode(text: str, vocab: list[str]) -> list[int]:
    encoded_text = [vocab.index(c) for c in text]
    return encoded_text


def decode(encoded_text: list[int], vocab: list[str]) -> str:
    decoded_text = "".join([vocab[c] for c in encoded_text])
    return decoded_text
