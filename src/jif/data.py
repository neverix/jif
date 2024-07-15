from datasets import load_dataset


def get_data(split="train"):
    for text in load_dataset("roneneldan/TinyStories", split=split):
        yield list(map(ord, text["text"]))