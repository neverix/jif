import json
import random

import requests
import torch
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, IterableDataset


class TSData(IterableDataset):
    def __init__(self, split="train",
                    seq_len=256,
                    epochs=None,
                    n_tokens=32_500,
                 ):
        self.seq_len = seq_len
        self.epochs = epochs
        self.base_seed = random.randint(0, 2**31)

        tokenizer_url = "https://huggingface.co/roneneldan/TinyStories-1M/raw/main/tokenizer.json"
        tokenizer_json = requests.get(tokenizer_url).json()
        tokenizer_json["added_tokens"][0]["id"] = n_tokens
        vocab = {k: v for k, v in tokenizer_json["model"]["vocab"].items() if v < n_tokens}
        vocab[tokenizer_json["added_tokens"][0]["content"]] = n_tokens
        tokenizer_json["model"]["vocab"] = vocab
        vocab = set(vocab.keys())
        tokenizer_json["model"]["merges"] = [v for v in tokenizer_json["model"]["merges"] if "".join(v.partition(" ")[::2]) in vocab]
        self.tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

        self.data = load_dataset("roneneldan/TinyStories", split=split)
        self.bos_token = self.tokenizer.token_to_id("<|endoftext|>")
        self.tokenizer.add_special_tokens(["<|pad|>"])
        self.pad_token = self.tokenizer.token_to_id("<|pad|>")
        self.n_classes = self.tokenizer.get_vocab_size()

    def __len__(self):
        # surely we will never run out of this much data!
        return len(self.data) * (self.epochs or 1e12)

    def __iter__(self):
        epochs = self.epochs
        base_seed = self.base_seed
        seq_len = self.seq_len
        tokenizer = self.tokenizer
        data = self.data
        pad_token = self.pad_token
        for epoch in (range(epochs) if epochs else iter(int, 1)):
            data.shuffle(seed=base_seed + epoch)
            for text in data:
                encoding = tokenizer.encode("<|endoftext|>" + text["text"])
                encoding.truncate(seq_len)
                encoding.pad(seq_len, pad_id=pad_token)
                yield torch.LongTensor(encoding.ids)


def get_data(batch_size, seq_len, split="train", epochs=None, n_tokens=2046):
    data = TSData(split=split, seq_len=seq_len, epochs=epochs, n_tokens=n_tokens)
    worker_seed = random.randrange(0, 2**32)
    generator_seed = random.randrange(0, 2**32)
    def data_generator():
        def seed_worker(worker_id):
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(generator_seed)
        return DataLoader(data, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=2,
                          worker_init_fn=seed_worker,
                          generator=g,)
    def detokenize(x):
        return data.tokenizer.decode_batch(x)
    return data_generator, detokenize, data.n_classes, data.bos_token
