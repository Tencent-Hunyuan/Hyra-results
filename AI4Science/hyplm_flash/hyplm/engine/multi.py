"""Data-parallel encoding over all visible GPUs: one engine process per device, length-balanced shards."""

import torch
import torch.multiprocessing as mp


def _worker(rank, device, checkpoint, precision, sequences, pooling, batch_tokens, queue):
    torch.cuda.set_device(device)
    from .model import HyPLMEngine
    engine = HyPLMEngine.from_pretrained(checkpoint, device=f"cuda:{device}", precision=precision)
    vectors = engine.encode(sequences, pooling=pooling, batch_tokens=batch_tokens)
    # A pickled array copy, so the worker may exit before the parent reads it (shared-memory handles may not).
    merged = torch.cat(vectors) if pooling == "residue" else torch.stack(vectors)
    queue.put((rank, merged.numpy()))


def encode_multi_gpu(sequences, checkpoint="weights", precision="int8", pooling="mean", batch_tokens=32768,
                     devices=None):
    """Encode on every GPU in parallel; returns CPU tensors in input order (see HyPLMEngine.encode)."""
    devices = list(range(torch.cuda.device_count())) if devices is None else list(devices)
    if len(devices) <= 1:
        from .model import HyPLMEngine
        engine = HyPLMEngine.from_pretrained(checkpoint, device=f"cuda:{devices[0] if devices else 0}", precision=precision)
        return engine.encode(sequences, pooling=pooling, batch_tokens=batch_tokens)
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))
    shards = [order[r::len(devices)] for r in range(len(devices))]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_worker, daemon=True,
                                 args=(r, d, str(checkpoint), precision, [sequences[i] for i in shards[r]],
                                       pooling, batch_tokens, queue))
                 for r, d in enumerate(devices)]
    for p in processes:
        p.start()
    results = [None] * len(sequences)
    for _ in processes:
        rank, merged = queue.get()
        merged = torch.from_numpy(merged)
        if pooling == "residue":
            pieces = merged.split([len(sequences[i]) for i in shards[rank]])
        else:
            pieces = merged.unbind(0)
        for index, piece in zip(shards[rank], pieces):
            results[index] = piece.clone()
    for p in processes:
        p.join()
        if p.exitcode:
            raise RuntimeError(f"encoding worker failed with exit code {p.exitcode}")
    return results
