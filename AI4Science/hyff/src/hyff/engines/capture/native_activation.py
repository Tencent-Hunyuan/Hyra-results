"""Native-activation EFM term as an additional compile choice of a checked EFMState.

Mix NativeActivationMixin into a checked EFMState class; nothing is installed
automatically. The state's _checked_compile self-check alone decides whether the
choice is accepted. Backend imports and environment checks run inside the choice
factory, so their failures reject only this choice.
"""
from __future__ import annotations


NATIVE_BACKEND_NAME = "native_activation"
_TAG = "_hyff_native_activation"


class NativeActivationUnavailable(RuntimeError):
    """Failure of the native-activation choice; the EFM compile check rejects that choice."""

    def __init__(self, stage, exception_type):
        self.stage, self.exception_type = stage, exception_type
        super().__init__(f"EFM native activation choice unavailable: {stage} ({exception_type})")


def is_tested_device(name):
    """True for the devices on which the native-activation backend was tested (NVIDIA H20)."""
    return name in ("H20", "NVIDIA H20")


def require_tested_environment(efm):
    # Raise unless torch, Triton and the GPU are the tested versions and device.
    # Imported lazily by the choice factory, not when constructing schedules.
    import torch

    device = next(efm.parameters()).device
    if device.type != "cuda":
        raise NativeActivationUnavailable("capability", "UnsupportedDevice")
    if torch.version.hip is not None:
        raise NativeActivationUnavailable("capability", "UnsupportedBackend")
    if torch.__version__.split("+", 1)[0] != "2.7.1":
        raise NativeActivationUnavailable("capability", "UntestedTorchVersion")
    if not is_tested_device(torch.cuda.get_device_name(device)):
        raise NativeActivationUnavailable("capability", "UntestedDevice")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise NativeActivationUnavailable("capability", "UntestedComputeCapability")
    import triton

    if triton.__version__ != "3.3.1":
        raise NativeActivationUnavailable("capability", "UntestedTritonVersion")
    from triton import language as tl

    if not callable(getattr(tl, "inline_asm_elementwise", None)):
        raise NativeActivationUnavailable("capability", "InlineAssemblyUnavailable")


def _normalize(stage, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except RuntimeError:
        raise
    except Exception as error:
        # Do not catch KeyboardInterrupt, SystemExit, or GeneratorExit.
        raise NativeActivationUnavailable(stage, type(error).__name__) from error


class NativeActivationMixin:
    """Adds the native-activation compile choice to a checked EFMState.

    Inherited checks run unchanged; only failures of this choice are normalized.

    Example integration (not executed here):
        class EFMState(NativeActivationMixin, ExistingCheckedEFMState):
            pass
    """

    def _checked_compile(self, index, efm):
        # Record the block index, then run the inherited checker unchanged.
        # Engine construction checks one block at a time on this state object.
        attribute = "_native_activation_check_index"
        missing = object()
        previous = getattr(self, attribute, missing)
        setattr(self, attribute, index)
        try:
            return super()._checked_compile(index, efm)
        finally:
            if previous is missing:
                delattr(self, attribute)
            else:
                setattr(self, attribute, previous)

    def _native_activation_capability(self, efm):
        require_tested_environment(efm)

    def _native_activation_builder(self):
        from hyff.engines.capture.native_activation_backend import make_term

        return make_term

    def _native_activation_factory(self, efm, tf32):
        if tf32:
            raise NativeActivationUnavailable("capability", "UnsupportedPrecision")
        _normalize("capability", self._native_activation_capability, efm)
        builder = _normalize("import", self._native_activation_builder)
        compiled = _normalize("factory", builder, efm)

        def term(*args, **kwargs):
            return _normalize("forward", compiled, *args, **kwargs)

        setattr(term, _TAG, True)
        if hasattr(compiled, "backend_stats"):
            term.backend_stats = compiled.backend_stats
        return term

    def _term_choices(self, efm):
        choices, reference = super()._term_choices(efm)
        # Offer the choice for blocks 1 and 2 only, never for a scalar-source block
        # (see the scalar-source backward note in efm_engine).
        if getattr(self, "_native_activation_check_index", None) not in (1, 2) or efm.input_is_scalar:
            return choices, reference
        names = [name for name, _ in choices]
        if "inductor" not in names or NATIVE_BACKEND_NAME in names:
            return choices, reference
        choice = (NATIVE_BACKEND_NAME, lambda tf32: self._native_activation_factory(efm, tf32))
        updated = list(choices)
        updated.insert(names.index("inductor"), choice)
        return updated, reference

    def _run_check(self, term_fn, *args, **kwargs):
        original = super()._run_check
        if not getattr(term_fn, _TAG, False):
            return original(term_fn, *args, **kwargs)
        # This covers the inherited check's autograd.grad call, including AOT's
        # lazy backward compilation, which runs after the forward wrapper has returned.
        return _normalize("forward_and_backward_check", original, term_fn, *args, **kwargs)
