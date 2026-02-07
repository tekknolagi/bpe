TRAINING_DATA = open("advs.txt").read()

import collections
import string

def merge_pair(data, pair, token):
    new_data = []
    i = 0
    while i < len(data):
        if i < len(data)-1 and (data[i], data[i+1]) == pair:
            new_data.append(token)
            i += 2
        else:
            new_data.append(data[i])
            i += 1
    return new_data

def train_one_round(tokens, token_ids, next_token_id, data, steps):
    if len(data) < 2:
        return data
    freq = collections.Counter()
    i = 0
    while i < len(data) - 1:
        freq[data[i], data[i+1]] += 1
        i += 1
    pair = freq.most_common(1)[0][0]
    steps.append(pair)
    token = pair[0] + pair[1]
    assert token not in tokens.values(), f"uh oh found {token!r} in {tokens}"
    tokens[next_token_id] = token
    token_ids[token] = next_token_id
    return merge_pair(data, pair, token)

def train(data, k=10):
    tokens = {x: chr(x) for x in range(256)}# if chr(x).isprintable()}
    token_ids = {v: k for k, v in tokens.items()}
    steps = []
    next_token_id = max(tokens.keys()) + 1
    for i in range(k):
        data = train_one_round(tokens, token_ids, next_token_id, data, steps)
        next_token_id += 1
    return data, tokens, token_ids, steps

def decode(tokens, data):
    result = ""
    for token_id in data:
        result += tokens[token_id]
    return result

RESET_ALL = "\033[0m"
RESET_FG = "\033[39m"
RESET_BG = "\033[49m"
FG_BLACK = "\033[30m"

BG_COLORS = [
    "\033[48;5;217m",
    "\033[48;5;158m",
    "\033[48;5;223m",
    "\033[48;5;183m",
    "\033[48;5;153m",
]

def colorize_tokens(tokens: list[str]) -> str:
    parts = []
    for i, token in enumerate(tokens):
        bg = BG_COLORS[i % len(BG_COLORS)]
        parts.append(f"{FG_BLACK}{bg}{token}{RESET_FG}{RESET_BG}")
    return "".join(parts) + RESET_ALL

def encode(token_ids, steps, text):
    data = list(text)
    for step in steps:
        data = merge_pair(data, step, step[0]+step[1])
    return [token_ids[t] for t in data]

import argparse
parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
subparsers = parser.add_subparsers(dest="command", required=True)
train_parser = subparsers.add_parser("train", help="Train a BPE tokenizer.")
train_parser.add_argument("--input", "-i", type=str, help="Input text file for training.")
train_parser.add_argument("--rounds", "-r", type=int, default=10, help="Number of training rounds.")
train_parser.add_argument("--output", "-o", type=str, help="Output config file.")
encode_parser = subparsers.add_parser("encode", help="Encode text using a trained BPE tokenizer.")
encode_parser.add_argument("--input", "-i", type=str, help="Input text file to encode.")
encode_parser.add_argument("--config", "-c", type=str, help="Trained config file.")
encode_parser.add_argument("--output", "-o", type=str, help="Output file to save the encoded data.")
decode_parser = subparsers.add_parser("decode", help="Decode text using a trained BPE tokenizer.")
decode_parser.add_argument("--input", "-i", type=str, help="Input text file to decode.")
decode_parser.add_argument("--config", "-c", type=str, help="Trained config file.")
decode_parser.add_argument("--output", "-o", type=str, help="Output file to save decoded data.")
args = parser.parse_args()

import json

if args.command == "train":
    k = args.rounds
    encoded, tokens, token_ids, steps = train(TRAINING_DATA, k)
    with open(args.output, "w+") as f:
        json.dump({"tokens": tokens, "steps": steps}, f)
elif args.command == "encode":
    with open(args.config, "r") as f:
        config = json.load(f)
    with open(args.input, "r") as f:
        token_ids = {v: int(k) for k, v in config["tokens"].items()}
        encoded = encode(token_ids, config["steps"], f.read())
    with open(args.output, "w+") as f:
        json.dump(encoded, f)
elif args.command == "decode":
    with open(args.config, "r") as f:
        config = json.load(f)
    with open(args.input, "r") as f:
        encoded = json.load(f)
    decoded = decode({int(k): v for k, v in config["tokens"].items()}, encoded)
    with open(args.output, "w+") as f:
        f.write(decoded)
else:
    raise "wtf"
