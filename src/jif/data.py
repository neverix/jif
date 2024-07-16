from more_itertools import chunked
from datasets import load_dataset


def get_data(split="train"):
    for text in load_dataset("roneneldan/TinyStories", split=split):
        yield list(map(ord, text["text"]))


def collate(generator, batch_size, seq_len, pad_token_id=0):
    for batch in chunked(generator, batch_size):
        batch = [text[:seq_len] for text in batch]
        lengths = [len(text) for text in batch]
        mask = [[1] * len(text) + [0] * (seq_len - len(text)) for text in batch]
        batch = [text + [pad_token_id] * (seq_len - len(text)) for text in batch]
        yield batch, lengths, mask