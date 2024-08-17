from datasets import load_dataset
from more_itertools import chunked
from tokenizers import Tokenizer
import random


def get_data(batch_size, seq_len, split="train", epochs=None):
    tokenizer = Tokenizer.from_pretrained("roneneldan/TinyStories-1M")
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
