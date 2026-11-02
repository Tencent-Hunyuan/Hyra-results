"""ESM-2's 33-token alphabet, without a network or Transformers dependency."""
from __future__ import annotations

from collections.abc import Sequence


class ESM2Tokenizer:
    """One amino acid per token; CLS/EOS are separate sequence boundaries.

    The non-ESM amino-acid ambiguity code J is normalized to X. Whitespace
    is stripped and lowercase is uppercased. Other unknown characters raise,
    so malformed FASTA headers cannot silently become protein sequences.
    """

    tokens = (
        "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E",
        "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W",
        "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>",
    )
    vocab_size = 33
    cls_token_id = bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    mask_token_id = 32
    special_token_ids = (0, 1, 2, 3, 31, 32)
    all_special_ids = special_token_ids
    standard_amino_acid_ids = tuple(range(4, 24))
    amino_acid_ids = tuple(range(4, 31))

    def __init__(self):
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        translation = bytearray([255] * 256)
        for token in self.tokens[4:31]:
            translation[ord(token)] = self.token_to_id[token]
        translation[ord("J")] = self.token_to_id["X"]
        self._translation = bytes(translation)

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def normalize(self, sequence: str | bytes) -> str:
        if isinstance(sequence, bytes):
            sequence = sequence.decode("ascii")
        if not isinstance(sequence, str):
            raise TypeError("sequence must be a string or ASCII bytes")
        sequence = "".join(sequence.split()).upper().replace("J", "X")
        invalid = set(sequence) - set(self.tokens[4:31])
        if invalid:
            raise ValueError(f"unsupported protein characters: {sorted(invalid)!r}")
        return sequence

    def encode(
        self,
        sequence: str | bytes,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation: bool = False,
    ) -> list[int]:
        return list(self.encode_bytes(sequence, add_special_tokens=add_special_tokens,
                                      max_length=max_length, truncation=truncation))

    def encode_bytes(
        self, sequence: str | bytes, add_special_tokens: bool = True,
        max_length: int | None = None, truncation: bool = False,
    ) -> bytes:
        """Same token IDs as ``encode`` using a C-level byte translation table."""
        if isinstance(sequence, str):
            sequence = "".join(sequence.split()).upper().encode("ascii")
        elif isinstance(sequence, bytes):
            sequence = b"".join(sequence.split()).upper()
        else:
            raise TypeError("sequence must be a string or ASCII bytes")
        normalized = sequence.translate(self._translation)
        if 255 in normalized:
            # The slower path constructs the useful error only for bad input.
            self.normalize(sequence)
            raise ValueError("unsupported protein bytes")
        boundary_count = 2 if add_special_tokens else 0
        if max_length is not None:
            if max_length < boundary_count:
                raise ValueError("max_length must leave room for sequence boundaries")
            if len(normalized) + boundary_count > max_length:
                if not truncation:
                    raise ValueError("sequence exceeds max_length; enable truncation explicitly")
                normalized = normalized[:max_length - boundary_count]
        return bytes([self.bos_token_id]) + normalized + bytes([self.eos_token_id]) if add_special_tokens else normalized

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        values = []
        for value in ids:
            value = int(value)
            if not 0 <= value < self.vocab_size:
                raise ValueError(f"token id outside vocabulary: {value}")
            if not skip_special_tokens or value not in self.special_token_ids:
                values.append(self.tokens[value])
        return "".join(values)

    def __call__(
        self,
        sequences: str | Sequence[str],
        *,
        padding: bool = True,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: str | None = "pt",
    ) -> dict:
        if isinstance(sequences, str):
            sequences = [sequences]
        encoded = [self.encode(s, max_length=max_length, truncation=truncation) for s in sequences]
        width = max(map(len, encoded), default=0)
        if return_tensors == "pt" and not padding and any(len(row) != width for row in encoded):
            raise ValueError("ragged tensors require padding=True")
        ids = [row + [self.pad_token_id] * (width - len(row)) if padding else row for row in encoded]
        mask = [[value != self.pad_token_id for value in row] for row in ids]
        if return_tensors is None:
            return {"input_ids": ids, "attention_mask": mask}
        if return_tensors != "pt":
            raise ValueError("return_tensors must be 'pt' or None")
        import torch
        return {"input_ids": torch.tensor(ids, dtype=torch.long).reshape(len(ids), width),
                "attention_mask": torch.tensor(mask, dtype=torch.bool).reshape(len(ids), width)}
