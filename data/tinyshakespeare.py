# modified from https://github.com/karpathy/llm.c/blob/master/dev/data/tinyshakespeare.py
"""
Downloads and tokenizes the TinyShakespeare dataset.
- The download is from Github.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created tinyshakespeare/ folder.
The script prints:

For GPT-2:
$ python dev/data/tinyshakespeare.py --model=gpt-2
writing 32,768 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin (66,560 bytes) in the gpt-2 format
writing 305,260 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin (611,544 bytes) in the gpt-2 format

For LLaMA 3:
$ python dev/data/tinyshakespeare.py --model=llama-3
writing 32,768 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin (132,096 bytes) in the llama-3 format
writing 276,224 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin (1,105,920 bytes) in the llama-3 format

And runs in a few seconds depending on your internet
connection and computer. The .bin files are raw byte
streams of uint16 (gpt-2) or uint32 (llama) numbers indicating the token ids.
"""

import os

import tiktoken

from data_common import download_file, write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare")


def download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    # download the TinyShakespeare dataset, unless it's already downloaded
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")


def tokenize():
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)
    eot = enc._special_tokens["<|endoftext|>"]  # end of text token
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    text = open(data_filename, "r").read()
    # let's treat every individual chunk of text as a separate "document"
    sections = text.split("\n\n")
    tokens = []
    for i, s in enumerate(sections):
        tokens.append(eot)
        # there was a mild bug where I originally intended to remove \n\n, but instead just added
        # the EOT right after each \n\n, so I'm keeping that behavior for backwards compatibility
        # therefore we have to here add an extra \n\n at the end of each section, except the last
        spad = s + "\n\n" if i != len(sections) - 1 else s
        tokens.extend(encode(spad))
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]
    # save to file
    val_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_tokens)
    write_datafile(train_filename, train_tokens)


if __name__ == "__main__":
    download()
    tokenize()
