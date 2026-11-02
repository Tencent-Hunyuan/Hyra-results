"""CPU checks of the latent-charge raw-charge bound (largo).

Small synthetic latent-charge frames isolate the charge, energy and force
arithmetic; no trained model, CUDA capture or multi-GPU path is involved.
"""
from __future__ import annotations

import copy
import statistics
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyff.models import factory as models
from hyff.models.charge_bound import (
    charge_bound_value, install_charge_bound,
    is_native_bound_conservation, normalize_charge_bound, raw_charges,
)
from hyff.models.latent_charges import LatentChargeEnergy

from helpers import bounded_charge_adapter as reference_hook  # noqa: E402


C = 6.824962139129639
CHARGES = "hyff-efm-charges-w672c1"
KEY = models.CHARGE_BOUND_KEY


class TinyMain(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_charge_energy = LatentChargeEnergy(8)
        self.latent_charge_spec = {"feature_width": 8}
        self.numerical_execution = {"policy": models.LAYERNORM_POLICY}


def tiny_main():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(413)
        model = TinyMain()
    with torch.no_grad():
        model.latent_charge_energy.charge_head.weight.copy_(torch.tensor([
            [-5.2, 3.1, 7.0, -2.1, 1.6, 4.3, -3.0, 2.4]
        ], dtype=torch.float32))
        model.latent_charge_energy.scale.fill_(0.7)
    return model


def frame(seed, grouped=False):
    generator = torch.Generator().manual_seed(seed)
    positions = (0.6 * torch.randn((7, 3), generator=generator)).requires_grad_(True)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1] if grouped else [0] * 7)
    total_charge = torch.tensor([1.0, -0.5] if grouped else [1.0], dtype=torch.float32)
    return positions, batch, total_charge


def scalars_from_positions(positions):
    # The scalar path depends on coordinates, so forces include the derivative
    # through the bound as well as through the native pair kernel.
    return torch.cat((positions, positions.square(),
                      positions.sin().sum(dim=1, keepdim=True),
                      positions.cos().sum(dim=1, keepdim=True)), dim=1)


def evaluate(model, seed, grouped=False):
    positions, batch, total_charge = frame(seed, grouped)
    scalars = scalars_from_positions(positions)
    term = model.latent_charge_energy
    head = raw_charges(term, scalars)
    charges = term.conserved_charges(scalars, batch, total_charge)
    energy = term(scalars, positions, batch, total_charge)
    force = -torch.autograd.grad(energy.sum(), positions)[0]
    return head.detach(), charges.detach(), energy.detach(), force.detach()


def cpu_measurements():
    """Return ordinary scalar measurements; no payloads or tensors are saved."""
    rows = []
    for seed, grouped in ((4, False), (5, True)):
        unbounded = tiny_main()
        adapted, native = copy.deepcopy(unbounded), copy.deepcopy(unbounded)
        handle = reference_hook.install_charge_bound_hook(adapted, c=C)
        install_charge_bound(native, C)
        reference, observed = evaluate(adapted, seed, grouped), evaluate(native, seed, grouped)
        original = evaluate(unbounded, seed, grouped)
        equal = [bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
                 for a, b in zip(reference, observed)]
        delta = [(a - b).double() for a, b in zip(observed, original)]
        rows.append({
            "seed": seed, "atoms": 7, "groups": 2 if grouped else 1,
            "native_vs_reference_hook_bitwise": dict(zip(("precenter_raw", "charge", "energy", "force"), equal)),
            "bound_vs_unbounded_precenter_raw_max": float(delta[0].abs().max()),
            "bound_vs_unbounded_charge_max": float(delta[1].abs().max()),
            "bound_vs_unbounded_energy_max_eV": float(delta[2].abs().max()),
            "bound_vs_unbounded_force_rms_eV_A": float(delta[3].square().mean().sqrt()),
            "bound_vs_unbounded_force_max_eV_A": float(delta[3].abs().max()),
            "finite": all(bool(torch.isfinite(value).all()) for value in observed),
        })
        handle.close()
    metrics = [key for key in rows[0] if key.startswith("bound_vs_unbounded_")]
    return {
        "device": "cpu", "dtype": "float32", "bound": C,
        "n": len(rows), "rows": rows,
        "summary": {key: {"n": len(rows),
                          "mean": statistics.mean(row[key] for row in rows),
                          "median": statistics.median(row[key] for row in rows)}
                    for key in metrics},
    }


class ChargeBoundTests(unittest.TestCase):
    def assertBitwise(self, left, right):
        self.assertEqual(left.dtype, torch.float32)
        self.assertEqual(right.dtype, torch.float32)
        self.assertTrue(torch.equal(left.view(torch.int32), right.view(torch.int32)))

    def test_missing_bound_is_unchanged_schema(self):
        config = {"hidden": 672, "atomic_numbers": [1, 6]}
        validated = models.validate_model_config(CHARGES, config)
        self.assertEqual(validated, config)
        self.assertIsNot(validated, config)
        self.assertNotIn(KEY, validated)

    def test_positive_plain_bound_normalized_to_fp32(self):
        for value in (C, 7, 0.1, 1e-30, 1e30):
            with self.subTest(value=value):
                actual = normalize_charge_bound(value)
                self.assertEqual(actual, float(torch.tensor(value, dtype=torch.float32)))
                self.assertEqual(models.validate_model_config(CHARGES, {
                    KEY: value})[KEY], actual)

    def test_malformed_bound_fails_closed(self):
        for value in (None, True, False, "6.824962139129639", [], {},
                      0, -1, 0.0, -0.0, -C, float("inf"), float("-inf"),
                      float("nan"), 1e-50, 1e50, 10 ** 1000):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, "charge bound"):
                    models.validate_model_config(CHARGES, {KEY: value})

    def test_bound_only_supported_on_latent_charge_constructors(self):
        for name in models.SUPPORTED_MODEL_IDS:
            with self.subTest(model_id=name):
                config = {KEY: C}
                if name.startswith("hyff-efm-charges-"):
                    self.assertEqual(models.validate_model_config(name, config), config)
                else:
                    with self.assertRaisesRegex(ValueError, "only by latent-charge constructors"):
                        models.validate_model_config(name, config)

    def test_factory_alias_bound_matches_named_model(self):
        canonical = models.factory_id(CHARGES)
        legacy = [old for old, new in models._LEGACY_MODEL_IDS.items() if new == CHARGES]
        self.assertEqual(len(legacy), 1)
        for alias in (CHARGES, canonical, *legacy):
            with self.subTest(alias=alias):
                self.assertEqual(models.validate_model_config(alias, {
                    KEY: C}), {KEY: C})

    def test_trunk_absent_marker_or_no_marker_loads(self):
        for model_id in models.SUPPORTED_MODEL_IDS:
            with self.subTest(model_id=model_id):
                self.assertEqual(models.validate_model_config(model_id, {}), {})
                self.assertEqual(models.validate_model_config(model_id, {
                    "trunk_dot_soft_cap": "ABSENT"}), {"trunk_dot_soft_cap": "ABSENT"})

    def test_any_present_trunk_cap_fails_closed(self):
        for value in (None, True, False, 0, 1, C, "PRESENT", "absent", "", {}, []):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, "trunk_dot_soft_cap is not supported in this release"):
                    models.validate_model_config(CHARGES, {"trunk_dot_soft_cap": value})

    def test_unknown_field_still_fails_closed(self):
        for value in ({KEY + "_typo": C}, {"unexpected": True}, {7: C}):
            with self.subTest(config=value):
                with self.assertRaisesRegex(ValueError, "outside the constructor schema"):
                    models.validate_model_config(CHARGES, value)

    def test_build_strips_artifact_fields_before_constructor(self):
        received = []
        model = tiny_main()
        def factory(values):
            received.append(dict(values))
            return model
        supplied = {"hidden": 672, KEY: C, "trunk_dot_soft_cap": "ABSENT"}
        with patch.object(models, "import_module", return_value=SimpleNamespace(build_w672c1=factory)):
            built = models.build_model({"format": models.FORMAT, "model_id": CHARGES,
                                        "model_config": supplied})
        self.assertIs(built, model)
        self.assertEqual(received, [{"hidden": 672}])
        self.assertEqual(supplied[KEY], C)
        self.assertEqual(charge_bound_value(built), C)

    def test_build_absent_bound_keeps_original_head_and_rng(self):
        model = tiny_main()
        head = model.latent_charge_energy.charge_head
        before = torch.get_rng_state().clone()
        with patch.object(models, "import_module", return_value=SimpleNamespace(build_w672c1=lambda _: model)):
            built = models.build_model({"model_id": CHARGES, "model_config": {}})
        self.assertIs(built.latent_charge_energy.charge_head, head)
        self.assertIs(type(head), nn.Linear)
        self.assertIsNone(charge_bound_value(built))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_native_install_preserves_parameters_buffers_state_and_rng(self):
        model = tiny_main().eval()
        parameters = [(name, id(value)) for name, value in model.named_parameters()]
        buffers = [(name, id(value)) for name, value in model.named_buffers()]
        state = {name: value.clone() for name, value in model.state_dict().items()}
        before = torch.get_rng_state().clone()
        install_charge_bound(model, C)
        self.assertEqual(parameters, [(name, id(value)) for name, value in model.named_parameters()])
        self.assertEqual(buffers, [(name, id(value)) for name, value in model.named_buffers()])
        self.assertEqual(list(state), list(model.state_dict()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(state[name], value))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(model.latent_charge_energy.charge_head.training)
        self.assertIs(type(model.latent_charge_energy.charge_head), nn.Linear)
        self.assertTrue(is_native_bound_conservation(model.latent_charge_energy))
        self.assertFalse(model.latent_charge_energy.charge_head._forward_hooks)
        self.assertFalse(model.latent_charge_energy.charge_head._forward_pre_hooks)
        for target in (model, model.latent_charge_energy):
            self.assertEqual(charge_bound_value(target), C)

    def test_install_twice_rejected(self):
        model = tiny_main()
        install_charge_bound(model, C)
        with self.assertRaisesRegex(ValueError, "already installed"):
            install_charge_bound(model, C)

    def test_install_invalid_native_heads_rejected(self):
        heads = (nn.Linear(8, 1), nn.Linear(8, 2, bias=False),
                 nn.Linear(8, 1, bias=False, dtype=torch.float64), nn.Identity(), None)
        for head in heads:
            with self.subTest(head_type=type(head).__name__):
                model = tiny_main()
                model.latent_charge_energy.charge_head = head
                with self.assertRaisesRegex(ValueError, "FP32 charge head"):
                    install_charge_bound(model, C)

    def test_install_rejects_prior_hooks_and_other_models(self):
        model = tiny_main()
        handle = model.latent_charge_energy.charge_head.register_forward_hook(lambda *_: None)
        with self.assertRaisesRegex(ValueError, "before charge-head observers"):
            install_charge_bound(model, C)
        handle.remove()
        with self.assertRaisesRegex(ValueError, "latent-charge"):
            install_charge_bound(nn.Linear(2, 1), C)

    def test_weight_state_can_load_strictly_between_native_and_reference_hook(self):
        source, bounded = tiny_main(), tiny_main()
        install_charge_bound(bounded, C)
        loaded = bounded.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        unbounded = tiny_main()
        loaded = unbounded.load_state_dict(bounded.state_dict(), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])

    def test_fp32_head_shape_and_dtype_are_enforced(self):
        model = tiny_main()
        install_charge_bound(model, C)
        with self.assertRaisesRegex(ValueError, r"\[N, 1\]"):
            raw_charges(model.latent_charge_energy, torch.ones(8))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "FP32 raw charges"):
                raw_charges(model.latent_charge_energy, torch.ones(3, 8))

    def test_bound_survives_charge_head_module_replacement(self):
        model = tiny_main()
        head = model.latent_charge_energy.charge_head
        install_charge_bound(model, C)
        self.assertIs(model.latent_charge_energy.charge_head, head)
        before = evaluate(model, 4)
        class ReplacementHead(nn.Module):
            def __init__(self, original):
                super().__init__()
                self.weight = original.weight
            def forward(self, values):
                return nn.functional.linear(values, self.weight)
        model.latent_charge_energy.charge_head = ReplacementHead(head)
        self.assertEqual(charge_bound_value(model.latent_charge_energy), C)
        for left, right in zip(before, evaluate(model, 4)):
            self.assertBitwise(left, right)
        unbounded = tiny_main()
        self.assertFalse(torch.equal(before[2], evaluate(unbounded, 4)[2]))

    def test_reference_hook_charge_energy_force_bitwise(self):
        for seed, grouped in ((4, False), (5, True)):
            with self.subTest(seed=seed, grouped=grouped):
                native, adapted = tiny_main(), tiny_main()
                install_charge_bound(native, C)
                handle = reference_hook.install_charge_bound_hook(adapted, c=C)
                try:
                    for actual, expected in zip(evaluate(native, seed, grouped), evaluate(adapted, seed, grouped)):
                        self.assertBitwise(actual, expected)
                        self.assertTrue(bool(torch.isfinite(actual).all()))
                finally:
                    handle.close()

    def test_bound_is_active_and_centers_after_transform(self):
        model = tiny_main()
        install_charge_bound(model, C)
        positions, batch, total = frame(4)
        scalars = scalars_from_positions(positions)
        term = model.latent_charge_energy
        raw = nn.functional.linear(scalars, term.charge_head.weight).squeeze(-1)
        bounded = C * torch.tanh(raw / C)
        sums = bounded.new_zeros(1).index_add(0, batch, bounded)
        expected = bounded + (total - sums) / len(bounded)
        self.assertBitwise(term.conserved_charges(scalars, batch, total), expected)
        before = raw + (total - raw.sum()) / len(raw)
        self.assertFalse(torch.equal(expected, C * torch.tanh(before / C)))
        self.assertTrue(bool((raw.abs() > C).any()))

    def test_measurement_scope_and_difference_are_explicit(self):
        result = cpu_measurements()
        self.assertEqual(result["n"], 2)
        for row in result["rows"]:
            self.assertTrue(all(row["native_vs_reference_hook_bitwise"].values()))
            self.assertTrue(row["finite"])
            self.assertGreater(row["bound_vs_unbounded_force_rms_eV_A"], 0)
            self.assertGreater(row["bound_vs_unbounded_energy_max_eV"], 0)
        self.assertTrue(all({"n", "mean", "median"} == set(value)
                            for value in result["summary"].values()))


if __name__ == "__main__":
    unittest.main()
