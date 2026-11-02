"""Pack variable-length proteins into one padding-free token stream for FlashAttention varlen."""

from dataclasses import dataclass

import numpy as np
import torch

from ..tokenizer import ESM2Tokenizer

TOKENIZER = ESM2Tokenizer()


def tokenize(sequence):
    """uint8 token ids including CLS and EOS."""
    return np.frombuffer(TOKENIZER.encode_bytes(sequence), dtype=np.uint8)


@dataclass
class Packed:
    ids: torch.Tensor          # [T] int64 token ids
    positions: torch.Tensor    # [T] int32 position within its protein (CLS = 0)
    cu_seqlens: torch.Tensor   # [B + 1] int32 segment offsets
    lengths: list              # tokens per protein, including CLS/EOS
    max_seqlen: int

    @property
    def tokens(self):
        return int(sum(self.lengths))


def pack(token_arrays, device, pad_to=None, max_segments=None, filler_segment=128):
    """Concatenate token arrays; optionally pad to a static size with short filler segments.

    Padding tokens (PAD id) form their own segments of at most `filler_segment` tokens so they
    never attend to real proteins and cost little attention work. Unused cu_seqlens slots repeat
    the final offset (zero-length segments), which FlashAttention skips.
    """
    lengths = [len(t) for t in token_arrays]
    total = sum(lengths)
    ids = np.concatenate(token_arrays).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    positions = (np.arange(total) - np.repeat(starts[:-1], lengths)).astype(np.int32)
    cu = list(starts)
    if pad_to is not None:
        if pad_to < total:
            raise ValueError("pad_to smaller than packed tokens")
        extra = pad_to - total
        ids = np.concatenate([ids, np.full(extra, TOKENIZER.pad_token_id, dtype=np.int64)])
        filler = np.arange(extra) % filler_segment
        positions = np.concatenate([positions, filler.astype(np.int32)])
        cu += list(range(total + filler_segment, pad_to, filler_segment)) + ([pad_to] if extra else [])
    if max_segments is not None:
        if len(cu) - 1 > max_segments:
            raise ValueError("too many segments for the static buffer")
        cu += [cu[-1]] * (max_segments + 1 - len(cu))
    pin = device.type == "cuda"
    to = lambda array: torch.from_numpy(array).pin_memory().to(device, non_blocking=True) if pin else torch.from_numpy(array)
    return Packed(to(ids), to(positions), to(np.asarray(cu, dtype=np.int32)), lengths,
                  max(lengths) if pad_to is None else max(max(lengths), filler_segment))


def plan(lengths, batch_tokens):
    """Groups of indices whose token counts (residues + 2) fit batch_tokens, in O(n).

    Each batch takes the longest remaining proteins, then tops up with the shortest ones.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    lo, hi, batches = 0, len(order) - 1, []
    while lo <= hi:
        batch, size = [], 0
        while lo <= hi and size + lengths[order[lo]] + 2 <= batch_tokens:
            batch.append(order[lo])
            size += lengths[order[lo]] + 2
            lo += 1
        while lo <= hi and size + lengths[order[hi]] + 2 <= batch_tokens:
            batch.append(order[hi])
            size += lengths[order[hi]] + 2
            hi -= 1
        if not batch:
            batch.append(order[lo])
            lo += 1
        batches.append(batch)
    return batches
