"""Protein embeddings and masked-LM logits from HyPLM-Flash on the GPU inference engine."""

import torch

from .engine.model import MAX_RESIDUES, HyPLMEngine
from .tokenizer import ESM2Tokenizer


class HyPLMEncoder:
    """HyPLM-Flash on a CUDA GPU (FlashAttention-3 on Hopper GPUs such as the H20, FlashAttention-2 elsewhere).

    precision="int8" (default) runs the projections on INT8 tensor cores; precision="bf16" keeps them in BF16
    and is recommended for supervised regression on near-identical variants.
    """

    max_residues = MAX_RESIDUES

    def __init__(self, engine):
        self.engine = engine
        self.tokenizer = ESM2Tokenizer()

    @classmethod
    def from_pretrained(cls, checkpoint="weights", device="cuda", precision="int8"):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("HyPLM-Flash runs on a CUDA GPU")
        return cls(HyPLMEngine.from_pretrained(checkpoint, device=device, precision=precision))

    def encode(self, sequences, pooling="residue", layer=None, batch_tokens=32768):
        """Embeddings as a list of CPU FP32 tensors in input order: [residues, 1280], or [1280] for pooling="mean".

        `layer` returns the output of an intermediate block (0 is the token embedding, 32 the final layer) instead
        of the final normalized representation. Proteins are packed into batches of up to `batch_tokens` tokens.
        """
        if pooling not in {"residue", "mean"}:
            raise ValueError("pooling must be residue or mean")
        if layer is not None and not 0 <= layer <= self.engine.num_layers:
            raise ValueError(f"layer must be between 0 and {self.engine.num_layers}")
        if isinstance(sequences, str):
            sequences = [sequences]
        return self.engine.encode(sequences, pooling=pooling, layer=layer, batch_tokens=batch_tokens)

    def logits(self, sequences, masks=None, batch_tokens=32768):
        """Masked-LM logits as a list of CPU FP32 tensors [residues, 33], in the order of tokenizer.tokens.

        masks[i] lists the 0-based positions of sequence i that are replaced by <mask> before the forward pass.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            masks = None if masks is None else [masks]
        if masks is not None and len(masks) != len(sequences):
            raise ValueError("masks must give one list of positions per sequence")
        return self.engine.logits(sequences, masks=masks, batch_tokens=batch_tokens)
