import fire

from .data import collate, get_data
from .diffusion import AbsorbingDiffusion


def main(
    batch_size = 4,
    seq_len = 128,
    n_classes = 128    
):
    diffusion = AbsorbingDiffusion(n_classes, 1e-3)
    for sample, _, _ in collate(get_data(), 4, 16):
        pass


if __name__ == "__main__":
    fire.Fire(main)
