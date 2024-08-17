from datasets import load_dataset
from more_itertools import chunked


def get_data(split="train"):
    def data_generator():
        for text in load_dataset("roneneldan/TinyStories", split=split):
            yield [0] + list(text["text"].encode("utf-8"))
    def detokenize(x):
        if isinstance(x, list) and not isinstance(x[0], int):
            return list(map(detokenize, x))
        return bytes([min(c, 255) for c in x])
    return detokenize, data_generator


def collate(generator_fn, batch_size, seq_len, pad_token_id=1, epochs=None):
    for _ in (range(epochs) if epochs is not None else iter(int, 1)):
        for batch in chunked(generator_fn(), batch_size):
            if len(batch) < batch_size:
                break
            batch = [text[:seq_len] for text in batch]
            lengths = [len(text) for text in batch]
            mask = [[1] * len(text) + [0] * (seq_len - len(text)) for text in batch]
            batch = [text + [pad_token_id] * (seq_len - len(text)) for text in batch]
            yield batch, lengths, mask

