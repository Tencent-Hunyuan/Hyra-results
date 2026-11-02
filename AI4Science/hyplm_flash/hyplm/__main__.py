"""Usage: python -m hyplm MKTAYIAKQRQISFVKSHFSRQ --output embeddings.pt"""

import argparse

import torch

from .inference import HyPLMEncoder


def main():
    parser = argparse.ArgumentParser(description="Extract HyPLM-Flash protein embeddings")
    parser.add_argument("sequences", nargs="+", help="protein sequences")
    parser.add_argument("--checkpoint", default="weights")
    parser.add_argument("--device", default="cuda", help="cuda or cuda:N")
    parser.add_argument("--precision", choices=["int8", "bf16"], default="int8")
    parser.add_argument("--pooling", choices=["residue", "mean"], default="residue")
    parser.add_argument("--output", default="embeddings.pt")
    args = parser.parse_args()
    encoder = HyPLMEncoder.from_pretrained(args.checkpoint, device=args.device, precision=args.precision)
    vectors = encoder.encode(args.sequences, pooling=args.pooling)
    torch.save(vectors, args.output)
    print(f"Saved {len(vectors)} tensors to {args.output}: {[tuple(vector.shape) for vector in vectors]}")


if __name__ == "__main__":
    main()
