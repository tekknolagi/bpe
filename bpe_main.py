if __name__ == "__main__":
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

    import bpe
    import pickle

    if args.command == "train":
        k = args.rounds
        data = open(args.input).read()
        encoded, tokens, token_ids, steps = bpe.train(data, k)
        with open(args.output, "wb+") as f:
            print(tokens)
            pickle.dump({"tokens": tokens, "steps": steps}, f)
    elif args.command == "encode":
        with open(args.config, "rb") as f:
            config = pickle.load(f)
        token_ids = {v: k for k, v in config["tokens"].items()}
        with open(args.input, "rb") as f:
            encoded = bpe.encode(token_ids, config["steps"], f.read())
        with open(args.output, "wb+") as f:
            pickle.dump(encoded, f)
    elif args.command == "decode":
        with open(args.config, "rb") as f:
            config = pickle.load(f)
        with open(args.input, "rb") as f:
            encoded = pickle.load(f)
        decoded = bpe.decode(config["tokens"], encoded)
        with open(args.output, "wb+") as f:
            f.write(decoded)
    else:
        raise "wtf"

