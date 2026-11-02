"""HyPLM inference engine for Hopper GPUs: packed variable-length batches, fused Triton kernels, INT8 projections.

The residual stream stays in FP32, and norms and rotary embeddings are computed in FP32. Precisions:
  int8  W8A8 projections on INT8 tensor cores, activations scaled per (token, 128 columns) and weights per
        output channel (default)
  bf16  BF16 projections, for supervised regression on near-identical variants
Below `quant_min_tokens` tokens per batch the projections run in BF16, where INT8 brings no speedup.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from .kernels.int8_gemm import Int8Weight, int8_gemm
from .kernels.norm import add_rmsnorm
from .kernels.quant import quantize_rows
from .kernels.rope import qk_norm_rope_, rope_tables
from .kernels.swiglu import swiglu
from .packing import TOKENIZER, pack, plan, tokenize

try:
    from flash_attn_interface import flash_attn_varlen_func as _fa3
except ImportError:
    _fa3 = None
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func as _fa2_packed
except ImportError:
    _fa2_packed = None

MAX_RESIDUES = 8190  # rotary tables cover 8192 positions, CLS and EOS included
PROJECTIONS = {"qkv": "mixer.qkv", "out": "mixer.out", "up": "mlp.up_gate", "down": "mlp.down"}


class HyPLMEngine:
    eps = 1e-6

    def __init__(self, state, config, device="cuda", precision="int8", quant_min_tokens=1024):
        if precision not in {"int8", "bf16"}:
            raise ValueError("precision must be int8 or bf16")
        if _fa3 is None and _fa2_packed is None:
            raise ImportError("the HyPLM engine needs FlashAttention (flash_attn_interface or flash_attn)")
        self.config = config
        self.device = device = torch.device(device)
        self.precision = precision
        self.num_layers = config["num_layers"]
        self.hidden = config["hidden_size"]
        self.heads = config["num_heads"]
        self.head_dim = self.hidden // self.heads
        self.quant_min_tokens = quant_min_tokens
        f32 = lambda name: state[name].to(device, torch.float32).contiguous()
        bf16 = lambda name: state[name].to(device, torch.bfloat16).contiguous()
        self.embedding = f32("embeddings.weight")
        self.layers = []
        for i in range(self.num_layers):
            p = f"layers.{i}."
            layer = {"norm1": f32(p + "norm1.weight"), "norm2": f32(p + "norm2.weight"),
                     "q_norm": f32(p + "mixer.q_norm.weight"), "k_norm": f32(p + "mixer.k_norm.weight")}
            for key, name in PROJECTIONS.items():
                layer[key] = bf16(p + name + ".weight")
                if precision == "int8":
                    layer[key + "_q"] = Int8Weight(state[p + name + ".weight"].to(device))
            self.layers.append(layer)
        self.final_norm = f32("final_norm.weight")
        self.head = {"dense_w": bf16("lm_head.dense.weight"), "dense_b": bf16("lm_head.dense.bias"),
                     "ln_w": f32("lm_head.layer_norm.weight"), "ln_b": f32("lm_head.layer_norm.bias"),
                     "dec_w": bf16("lm_head.decoder.weight"), "dec_b": bf16("lm_head.decoder.bias")}
        self.head_eps = config["mlm_norm_eps"]
        self.cos, self.sin = rope_tables(self.head_dim, config["rope_theta"], MAX_RESIDUES + 2, device)

    @classmethod
    def from_pretrained(cls, checkpoint="weights", device="cuda", precision="int8", **kwargs):
        checkpoint = Path(checkpoint)
        config = json.loads((checkpoint / "config.json").read_text())
        return cls(load_file(checkpoint / "model.safetensors", device="cpu"), config, device, precision, **kwargs)

    def _attention(self, qkv, packed):
        tokens = qkv.shape[0]
        if _fa3 is not None:
            q, k, v = qkv.unbind(1)
            out = _fa3(q, k, v, packed.cu_seqlens, packed.cu_seqlens, packed.max_seqlen, packed.max_seqlen)
            out = out[0] if isinstance(out, tuple) else out
        else:
            out = _fa2_packed(qkv, packed.cu_seqlens, packed.max_seqlen)
        return out.reshape(tokens, self.hidden).to(torch.bfloat16)

    def _project(self, w, name, h):
        """h: BF16 activations, or (INT8 values, scales) produced by a fused kernel."""
        if isinstance(h, tuple):
            return int8_gemm(h[0], h[1], w[name + "_q"])
        return F.linear(h, w[name])

    @torch.inference_mode()
    def forward_packed(self, packed, layers=(), final_dtype=torch.float32):
        """Run all blocks on a packed stream; return (final normed states [T, H], {layer: [T, H]})."""
        x = self.embedding[packed.ids]
        tokens = x.shape[0]
        saved = {0: x.clone()} if 0 in layers else {}
        quant = self.precision == "int8" and tokens >= self.quant_min_tokens
        delta = None
        for i, w in enumerate(self.layers):
            h = add_rmsnorm(x, delta, w["norm1"], self.eps, quant=quant)
            if i in layers and i > 0:
                saved[i] = x.clone()
            qkv = self._project(w, "qkv", h)
            qkv = qk_norm_rope_(qkv.view(tokens, 3, self.heads, self.head_dim), packed.positions,
                                w["q_norm"], w["k_norm"], self.cos, self.sin, self.eps)
            o = self._attention(qkv, packed)
            a = self._project(w, "out", quantize_rows(o) if quant else o)
            h = add_rmsnorm(x, a, w["norm2"], self.eps, quant=quant)
            delta = self._project(w, "down", swiglu(self._project(w, "up", h), quant=quant))
        y = add_rmsnorm(x, delta, self.final_norm, self.eps, out_dtype=final_dtype)
        if self.num_layers in layers:
            saved[self.num_layers] = y
        return y, saved

    @torch.inference_mode()
    def lm_logits(self, states):
        """MLM head on final states [n, H]."""
        w = self.head
        h = F.gelu(F.linear(states.to(torch.bfloat16), w["dense_w"], w["dense_b"]))
        h = F.layer_norm(h.float(), (h.shape[-1],), w["ln_w"], w["ln_b"], self.head_eps)
        return F.linear(h.to(torch.bfloat16), w["dec_w"], w["dec_b"])

    # ---- APIs on protein strings ----
    def _normalize(self, sequences):
        sequences = [TOKENIZER.normalize(s) for s in sequences]
        if any(not s or len(s) > MAX_RESIDUES for s in sequences):
            raise ValueError(f"sequences must have 1 to {MAX_RESIDUES} residues")
        return sequences

    def pack(self, sequences, **kwargs):
        return pack([tokenize(s) for s in sequences], self.device, **kwargs)

    @torch.inference_mode()
    def hidden_states(self, sequences, layers):
        """Per sequence {layer: [len, H]} residue-aligned (CLS/EOS removed); layer 0 = embeddings."""
        packed = self.pack(self._normalize(sequences))
        _, saved = self.forward_packed(packed, layers=set(layers))
        starts = np.concatenate([[0], np.cumsum(packed.lengths)])
        return [{k: saved[k][a + 1:a + n - 1] for k in layers} for a, n in zip(starts, packed.lengths)]

    @torch.inference_mode()
    def masked_logits(self, ids, mask, rows, cols):
        """Logits at (row, col) of a right-padded batch [B, L] (already masked), via packing."""
        lengths = mask.sum(1).tolist()
        packed = pack([ids[i, :n].cpu().numpy() for i, n in enumerate(lengths)], self.device)
        y, _ = self.forward_packed(packed)
        return self.lm_logits(y[packed.cu_seqlens[rows].long() + cols])

    @torch.inference_mode()
    def encode(self, sequences, pooling="residue", layer=None, batch_tokens=32768, dtype=torch.float32):
        """List of CPU tensors per protein: [len, H] residue states or [H] mean over residues.

        `layer` selects the output of an intermediate block (default: final normed states).
        """
        sequences = self._normalize(sequences)
        results = [None] * len(sequences)
        for batch in plan([len(s) for s in sequences], batch_tokens):
            packed = self.pack([sequences[i] for i in batch])
            y, saved = self.forward_packed(packed, layers={layer} if layer is not None else ())
            states = y if layer is None else saved[layer]
            if pooling == "mean":
                seg = torch.repeat_interleave(torch.arange(len(batch), device=self.device),
                                              torch.tensor(packed.lengths, device=self.device))
                keep = (packed.positions > 0) & (packed.positions < (packed.cu_seqlens[seg + 1] - packed.cu_seqlens[seg] - 1))
                sums = torch.zeros(len(batch), self.hidden, device=self.device).index_add_(0, seg[keep], states[keep].float())
                counts = torch.tensor([n - 2 for n in packed.lengths], device=self.device, dtype=torch.float32)
                pooled = (sums / counts[:, None]).to(dtype).cpu()
                for row, index in enumerate(batch):
                    results[index] = pooled[row]
            else:
                host = states.to(dtype).cpu()
                start = 0
                for index, n in zip(batch, packed.lengths):
                    results[index] = host[start + 1:start + n - 1]
                    start += n
        return results

    @torch.inference_mode()
    def logits(self, sequences, masks=None, batch_tokens=32768):
        """MLM logits [len, 33] per protein (CPU, FP32); the residues at masks[i] are replaced by <mask> first."""
        sequences = self._normalize(sequences)
        results = [None] * len(sequences)
        for batch in plan([len(s) for s in sequences], batch_tokens):
            arrays = []
            for i in batch:
                tokens = tokenize(sequences[i]).copy()
                if masks is not None and len(masks[i]):
                    tokens[np.asarray(masks[i], dtype=np.int64) + 1] = TOKENIZER.mask_token_id
                arrays.append(tokens)
            packed = pack(arrays, self.device)
            y, _ = self.forward_packed(packed)
            out = self.lm_logits(y).float().cpu()
            start = 0
            for index, n in zip(batch, packed.lengths):
                results[index] = out[start + 1:start + n - 1]
                start += n
        return results
