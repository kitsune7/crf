import sys

from crf import data_loader


def main():
    sentences = data_loader.load_data_split()
    print(sentences["train"][0])
    print(sentences["test"][0])
    print(sentences["validation"][0])


if __name__ == "__main__":
    sys.exit(main())

