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

def train(text, k=10):
    text = text.encode("utf-8")
    tokens = {}
    seen_tokens = set()
    # Give the lowest token IDs to the chars in the corpus
    for c in text:
        c = bytes([c])
        if c not in seen_tokens:
            tokens[len(tokens)] = c
            seen_tokens.add(c)
    token_ids = {v: k for k, v in tokens.items()}
    steps = []
    next_token_id = max(tokens.keys()) + 1
    data = [bytes([c]) for c in text]
    for i in range(k):
        data = train_one_round(tokens, token_ids, next_token_id, data, steps)
        next_token_id += 1
    for i in range(256):
        c = bytes([i])
        if c not in seen_tokens:
            tokens[len(tokens)] = c
            seen_tokens.add(c)
    return data, tokens, token_ids, steps

def decode(tokens, data):
    result = b""
    for token_id in data:
        result += tokens[token_id]
    return result

def encode(token_ids, steps, text):
    data = [bytes([c]) for c in text]
    for step in steps:
        data = merge_pair(data, tuple(step), step[0]+step[1])
    return [token_ids[t] for t in data]

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
