from .data import get_data, collate
import fire


def main():
    for sample in collate(get_data(), 4, 16):
        print(sample)
        break


if __name__ == "__main__":
    fire.Fire(main)
