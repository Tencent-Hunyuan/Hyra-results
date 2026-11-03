"""Measure saved-tensor storage of the replay VJP itself; no extra edge execution or tape.

Only storage identities/sizes and scalar counts survive a call. The hooks cover
only the replay forward; parity probes, the retained-graph backward and compiler
bookkeeping are not counted.
"""
from __future__ import annotations


class ReplayStorageObservation:
    def __init__(self, efm):
        self._excluded = set()
        for value in (*efm.parameters(), *efm.buffers()):
            key, _ = self._storage(value)
            if key is not None:
                self._excluded.add(key)
        self.samples = 0
        self.max_unique_retained_bytes = 0
        self.rows = None
        self.invalid_reason = None
        self.max_saved_hook_calls = 0
        self.max_unique_hook_storages = 0

    @staticmethod
    def _storage(value):
        if getattr(value, "fake_mode", None) is not None or value.device.type not in ("cpu", "cuda"):
            raise ValueError("EFM_CAPACITY_REAL_TENSOR_OBSERVATION_REQUIRED")
        if not value.numel():
            return None, 0
        storage = value.untyped_storage()
        return (value.device.type, value.device.index, storage.data_ptr()), storage.nbytes()

    def vjp(self, forward, leaves, cotangents, *, rows):
        import torch
        sizes, hook_keys = {}, set()
        calls = 0
        invalid = None

        def observe(value, *, hook=False):
            nonlocal invalid
            try:
                key, size = self._storage(value)
            except (ValueError, RuntimeError, TypeError):
                invalid = "NON_REAL_TENSOR_SAVE"
                return
            if key is not None and key not in self._excluded:
                sizes[key] = size
                if hook:
                    hook_keys.add(key)

        def pack(value):
            nonlocal calls
            calls += 1
            observe(value, hook=True)
            return value.detach()

        for value in leaves:
            observe(value)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            outputs = forward()
        for value in outputs:
            observe(value)
        gradients = torch.autograd.grad(outputs, leaves, cotangents,
                                        allow_unused=True, create_graph=False)
        # Publish only after the actual VJP succeeded. No tensor/graph escapes.
        self.samples += 1
        if invalid is not None or not hook_keys or not calls:
            self.invalid_reason = invalid or "NO_REAL_SAVED_TENSORS"
        if self.rows is not None and self.rows != rows:
            self.invalid_reason = "MIXED_TILE_ROWS"
        self.rows = rows
        self.max_unique_retained_bytes = max(self.max_unique_retained_bytes, sum(sizes.values()))
        self.max_saved_hook_calls = max(self.max_saved_hook_calls, calls)
        self.max_unique_hook_storages = max(self.max_unique_hook_storages, len(hook_keys))
        return gradients

    def public_metadata(self):
        valid = self.samples > 0 and self.invalid_reason is None and self.max_unique_retained_bytes > 0
        return dict(schema="EFM_ACTUAL_REPLAY_STORAGE", valid=valid, samples=self.samples,
                    rows=self.rows, unique_retained_bytes=self.max_unique_retained_bytes if valid else None,
                    saved_hook_calls=self.max_saved_hook_calls, unique_hook_storages=self.max_unique_hook_storages,
                    invalid_reason=self.invalid_reason if self.samples else "NO_ACTUAL_REPLAY_VJP",
                    scope="FIRST_BASELINE_ACTUAL_REPLAY_VJP_FORWARD_SAVES_PLUS_LEAVES_OUTPUTS",
                    max_over_observed_tiles=True, separate_calibration_force=False,
                    extra_edge_executions=0, live_tensors_retained=0, graph_returned=False,
                    excludes_first_use_parity_probe=True, shared_parameters_buffers_excluded=True,
                    leaves_outputs_included=True, full_storage_views_deduplicated=True)


__all__ = ["ReplayStorageObservation"]
