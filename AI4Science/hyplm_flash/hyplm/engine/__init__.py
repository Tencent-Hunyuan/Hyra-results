"""Optional high-throughput HyPLM inference engine for Hopper GPUs (needs triton and flash-attn)."""

__all__ = ["HyPLMEngine", "encode_multi_gpu"]


def __getattr__(name):
    # Lazy so that CPU-only code (e.g. hyplm.engine.packing) imports without a GPU driver.
    if name == "HyPLMEngine":
        from .model import HyPLMEngine
        return HyPLMEngine
    if name == "encode_multi_gpu":
        from .multi import encode_multi_gpu
        return encode_multi_gpu
    raise AttributeError(name)
