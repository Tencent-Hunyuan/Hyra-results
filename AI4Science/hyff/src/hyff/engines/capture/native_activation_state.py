"""largo EFM state with the native-activation backend as an extra choice; the self-check decides acceptance."""
from __future__ import annotations

from copy import deepcopy
import os
import time

from hyff.engines.capture.native_activation import (
    NATIVE_BACKEND_NAME, NativeActivationMixin, NativeActivationUnavailable,
)
from hyff.engines.capture.efm_port_grouped import EFMState as ExistingSegmentedEFMState


def native_activation_enabled():
    """On by default; HYFF_PRECISE_EFM=0 keeps the compiled-only schedule (for example for short jobs)."""
    value = os.environ.get('HYFF_PRECISE_EFM')
    if value is None or value == '1':
        return True
    if value == '0':
        return False
    raise ValueError('HYFF_PRECISE_EFM must be 0 or 1 when set')


class NativeActivationEFMState(NativeActivationMixin, ExistingSegmentedEFMState):
    """Segmented largo EFM state with the native-activation choice; compilation and checks are inherited."""

    def _record_native_activation_failure(self, error):
        """Record the failure stage and exception types once per error, without keeping the exception or traceback."""
        token = self.__dict__.setdefault('_native_activation_failure_observer_token', object())
        observed = error.__dict__.setdefault('_hyff_native_failure_observer_tokens', set())
        if token in observed:
            return
        cause = error.__cause__
        event = dict(
            block=getattr(self, '_native_activation_check_index', None),
            stage=error.stage,
            exception_type=error.exception_type,
            cause_type=None if cause is None else type(cause).__name__,
        )
        self.__dict__.setdefault('_native_activation_failure_events', []).append(event)
        observed.add(token)

    def _native_activation_factory(self, efm, tf32):
        try:
            return super()._native_activation_factory(efm, tf32)
        except NativeActivationUnavailable as error:
            self._record_native_activation_failure(error)
            raise

    def _run_check(self, term_fn, *args, **kwargs):
        try:
            return super()._run_check(term_fn, *args, **kwargs)
        except NativeActivationUnavailable as error:
            self._record_native_activation_failure(error)
            raise

    def _term_choices(self, efm):
        choices, reference = super()._term_choices(efm)
        index = getattr(self, '_native_activation_check_index', None)
        if index is not None:
            schedules = self.__dict__.setdefault('_native_activation_schedules', {})
            schedules.setdefault(index, []).append([name for name, _ in choices])
        return choices, reference

    def _checked_compile(self, index, efm):
        started = time.perf_counter()
        result = None
        try:
            result = super()._checked_compile(index, efm)
            return result
        finally:
            timings = self.__dict__.setdefault('_native_activation_build_timings', {})
            timings[index] = dict(
                wall_seconds=time.perf_counter() - started,
                selected=None if result is None else result[1],
                completed=result is not None,
                scope='whole_block_check_including_actual_and_nonzero_probe',
            )

    def _native_activation_builder(self):
        builder = super()._native_activation_builder()

        def build(efm):
            term = builder(efm)
            # The largo compile policy wraps torch.compile and overrides the
            # backend's dynamic=False request. Read the returned callable's
            # marker before the choice wrapper hides it.
            stats = getattr(term, 'backend_stats', None)
            if isinstance(stats, dict):
                dynamic_marker = getattr(term, 'hyff_compile_dynamic', None)
                stats['compile_flags'] = dict(
                    requested_dynamic=False,
                    effective_dynamic=dynamic_marker if type(dynamic_marker) is bool else None,
                    effective_dynamic_status='OBSERVED' if type(dynamic_marker) is bool else 'UNKNOWN',
                    effective_dynamic_source='compiled_callable.hyff_compile_dynamic',
                    fullgraph=True,
                    triton_cudagraphs=False,
                    force_disable_caches=True,
                )
            return term

        return build

    def report(self):
        record = super().report()
        selected = sorted(index for index, name in self.term_backend.items()
                          if name == NATIVE_BACKEND_NAME)
        record['native_activation'] = dict(
            name=NATIVE_BACKEND_NAME,
            enabled=True,
            opt_out='HYFF_PRECISE_EFM=0',
            eligible_blocks=[1, 2],
            selected_blocks=selected,
            build_times_by_block=deepcopy(getattr(self, '_native_activation_build_timings', {})),
            failure_events=deepcopy(getattr(self, '_native_activation_failure_events', [])),
            backend_stats={index: deepcopy(getattr(self.compiled_term[index], 'backend_stats', {}))
                           for index in selected},
        )
        return record
