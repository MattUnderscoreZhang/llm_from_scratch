def get_token_set(text: str) -> list[str]:
    chars = sorted(list(set(text)))
    return chars


def encode(text: str, token_set: list[str]) -> list[int]:
    encoded_text = [token_set.index(c) for c in text]
    return encoded_text


def decode(encoded_text: list[int], token_set: list[str]) -> str:
    decoded_text = "".join([token_set[c] for c in encoded_text])
    return decoded_text
