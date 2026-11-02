"""Reference batching: the plain PyTorch model with IEEE FP32 products, one evaluation per call."""
from hyff.batch.base import BatchedPotential


class ReferenceBatchedPotential(BatchedPotential):
    """Reuse the first eager result computed during entry preparation.

    Models with an electronic attention head compute their initial scalar, and
    install the per-block electronic context, during every raw forward. For
    those models the model's own bound ``initial_scalar`` is restored around each
    eager call; CUDA-graph capture is rejected for them.
    """
    def __init__(self, model, *args, **kwargs):
        attention = hasattr(model, '_attention_electronic_native_initial_scalar')
        if attention and kwargs.get('backend', 'cuda_graph') != 'eager':
            raise ValueError('Reference batching of models with electronic attention requires the eager backend')
        self._reference_original_initial_scalar = model.initial_scalar if attention else None
        super().__init__(model, *args, **kwargs)

    def _make_entry(self, values):
        entry = super()._make_entry(values)
        if self.backend == 'eager':
            entry['_first_eager_result_ready'] = True
        return entry

    def _compute(self, entry):
        if entry.pop('_first_eager_result_ready', False):
            return entry['energy'], entry['force']
        original = self._reference_original_initial_scalar
        if original is None:
            return super()._compute(entry)
        current = self.model.initial_scalar
        self.model.initial_scalar = original
        try:
            return super()._compute(entry)
        finally:
            self.model.initial_scalar = current
