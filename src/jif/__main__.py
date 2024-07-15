from .data import get_data
import fire


def main():
    for sample in get_data():
        print(sample)
        break


if __name__ == "__main__":
    fire.Fire(main)
