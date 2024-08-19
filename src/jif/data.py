from datasets import load_dataset
from more_itertools import chunked
from tokenizers import Tokenizer
import requests
import random
import json


def get_data(batch_size, seq_len, split="train", epochs=None, n_tokens=2046):
    tokenizer_url = "https://huggingface.co/roneneldan/TinyStories-1M/raw/main/tokenizer.json"
    tokenizer_json = requests.get(tokenizer_url).json()
    tokenizer_json["added_tokens"][0]["id"] = n_tokens
    vocab = {k: v for k, v in tokenizer_json["model"]["vocab"].items() if v < n_tokens}
    vocab[tokenizer_json["added_tokens"][0]["content"]] = n_tokens
    tokenizer_json["model"]["vocab"] = vocab
    vocab = set(vocab.keys())
    tokenizer_json["model"]["merges"] = [v for v in tokenizer_json["model"]["merges"] if "".join(v.partition(" ")[::2]) in vocab]
    tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

    bos_token = tokenizer.token_to_id("<|endoftext|>")
    tokenizer.add_special_tokens(["<|pad|>"])
    pad_token = tokenizer.token_to_id("<|pad|>")
    n_classes = tokenizer.get_vocab_size()
    def data_generator():
        base_seed = random.randint(0, 2**31)
        data = load_dataset("roneneldan/TinyStories", split=split)
        for epoch in (range(epochs) if epochs else iter(int, 1)):
            data.shuffle(seed=base_seed + epoch)
            for texts in chunked(data, batch_size):
                if len(texts) < batch_size:
                    continue
                encodings = tokenizer.encode_batch(["<|endoftext|>" + x["text"] for x in texts])
                for enc in encodings:
                    enc.truncate(seq_len)
                    enc.pad(seq_len, pad_id=pad_token)
                yield [enc.ids for enc in encodings]
    def detokenize(x):
        return tokenizer.decode_batch(x)
    return data_generator, detokenize, n_classes, bos_token
