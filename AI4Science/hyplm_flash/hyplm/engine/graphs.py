"""Whole-model CUDA graphs over static token buckets, removing launch overhead for small batches.

A packed batch is padded to the next bucket with short filler segments (see packing.pack); each
bucket owns static input buffers and one captured graph, and all graphs share one memory pool.
On H20 a forward pass is launch-bound only below ~400 tokens, so larger batches run eagerly.
"""

import torch

from .packing import Packed, pack, tokenize

BUCKETS = tuple(range(64, 513, 64))


class GraphRunner:
    def __init__(self, engine, buckets=BUCKETS, max_seqlen=2048, filler=64):
        self.engine = engine
        self.buckets = tuple(sorted(buckets))
        self.max_seqlen = max_seqlen
        self.filler = filler
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}

    def max_segments(self, bucket):
        return bucket // 8 + bucket // self.filler + 2

    def _capture(self, bucket):
        device = self.engine.device
        segments = self.max_segments(bucket)
        static = Packed(torch.full((bucket,), 1, dtype=torch.long, device=device),
                        torch.zeros(bucket, dtype=torch.int32, device=device),
                        torch.full((segments + 1,), bucket, dtype=torch.int32, device=device),
                        [bucket], self.max_seqlen)
        static.cu_seqlens[0] = 0
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.engine.forward_packed(static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out, _ = self.engine.forward_packed(static)
        self.graphs[bucket] = (graph, static, out)

    def bucket_for(self, tokens):
        for bucket in self.buckets:
            if tokens <= bucket:
                return bucket
        return None

    @torch.inference_mode()
    def forward(self, token_arrays):
        """Final normed states [bucket, H] (static buffer, valid until the next call) and the packing."""
        tokens = sum(len(t) for t in token_arrays)
        bucket = self.bucket_for(tokens)
        fillers = 0 if bucket is None else -(-(bucket - tokens) // self.filler)
        if bucket is None or len(token_arrays) + fillers > self.max_segments(bucket):
            packed = pack(token_arrays, self.engine.device)
            return self.engine.forward_packed(packed)[0], packed
        if bucket not in self.graphs:
            self._capture(bucket)
        graph, static, out = self.graphs[bucket]
        packed = pack(token_arrays, self.engine.device, pad_to=bucket, max_segments=self.max_segments(bucket),
                      filler_segment=self.filler)
        static.ids.copy_(packed.ids)
        static.positions.copy_(packed.positions)
        static.cu_seqlens.copy_(packed.cu_seqlens)
        graph.replay()
        return out, packed

    def encode_one(self, sequence):
        """Low-latency path: final normed residue states [L, H] for one protein (GPU tensor)."""
        out, _ = self.forward([tokenize(sequence)])
        return out[1:len(sequence) + 1]
