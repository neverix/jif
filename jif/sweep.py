import random
import numpy as np
from tqdm import tqdm
from .__main__ import train

def sweep(*args, min_lr, max_lr, out_name, **kwargs):
    from matplotlib import pyplot as plt
    from collections import defaultdict
    from itertools import product
    lrs = np.exp(np.linspace(np.log(min_lr), np.log(max_lr), 10))
    models = ["small", "medium", "big"]
    colors = ["r", "g", "b"]
    lrs_sampled = defaultdict(list)
    losses = defaultdict(list)
    all_configs = list(product(lrs, models))
    # random.Random(0).shuffle(all_configs)
    random.shuffle(all_configs)
    for lr, model in tqdm(all_configs):
        print("Training", model, "model with", lr, "learning rate")
        lrs_sampled[model].append(lr)
        log_dict = train(*args, **kwargs | {"n_steps": 1_000, "lr": lr, "size": model},
                         quiet=True, profile=False)
        losses[model].append(log_dict["loss_sma"])

        for model, color in zip(models, colors):
            plt.scatter(lrs_sampled[model], losses[model], label=model, c=color)
            if losses[model]:
                lr, loss = zip(*sorted(zip(lrs_sampled[model], losses[model]), key=lambda x: x[0]))
                plt.plot(lr, loss, c=color)
        plt.xlim(lrs[0], lrs[-1])
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Final loss")
        plt.legend()
        plt.savefig(f"{out_name}.png")
        plt.close()


if __name__ == "__main__":
    # sweep(min_lr=7e-4, max_lr=3e-3, out_name="lr_batch_search")
    sweep(min_lr=5e-4, max_lr=5e-3, out_name="lr_search", fix_batch_size=True)