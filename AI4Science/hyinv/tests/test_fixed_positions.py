import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hyinv import HyInv, Structure, design
from hyinv.pdb import parse_fixed, read_pdb


def pdb_line(serial, atom, resname, chain, number, insertion):
    return (
        f"ATOM  {serial:5d} {atom:>4s} {resname:>3s} {chain}{number:4d}{insertion:1s}   "
        f"{serial * 0.7:8.3f}{serial * 0.3:8.3f}{serial * 0.2:8.3f}"
        "  1.00 20.00           C\n"
    )


class FixedInsertionCodesTest(unittest.TestCase):
    def setUp(self):
        residues = [
            ("GLY", "A", -1, " "), ("ALA", "A", 12, " "),
            ("SER", "A", 12, "A"), ("TYR", "A", 12, "B"),
            ("THR", "A", 13, " "), ("VAL", "B", 12, "A"),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as pdb:
            self.addCleanup(Path(pdb.name).unlink, missing_ok=True)
            self.pdb_path = pdb.name
            serial = 0
            for name, chain, number, insertion in residues:
                for atom in ("N", "CA", "C", "O"):
                    serial += 1
                    pdb.write(pdb_line(serial, atom, name, chain, number, insertion))
        self.structure = read_pdb(pdb.name)

    def assert_fixed(self, spec, expected):
        np.testing.assert_array_equal(parse_fixed(spec, self.structure), expected)

    def test_insertions_and_legacy_number_selection(self):
        self.assert_fixed("A12", [False, True, True, True, False, False])
        self.assert_fixed("A12.", [False, True, False, False, False, False])
        self.assert_fixed("A12A,B12A", [False, False, True, False, False, True])
        self.assert_fixed("A12B", [False, False, False, True, False, False])
        self.assert_fixed("A-1", [True, False, False, False, False, False])
        single_chain = read_pdb(self.pdb_path, chains="A")
        np.testing.assert_array_equal(parse_fixed("12A", single_chain), [False, False, True, False, False])

    def test_inclusive_insertion_code_ranges(self):
        self.assert_fixed("A12A-12B", [False, False, True, True, False, False])
        self.assert_fixed("A12.-12A", [False, True, True, False, False, False])
        self.assert_fixed("A12A-12", [False, False, True, True, False, False])
        self.assert_fixed("A12-12A", [False, True, True, False, False, False])
        self.assert_fixed("A12.-12.", [False, True, False, False, False, False])
        self.assert_fixed("A12-13", [False, True, True, True, True, False])
        self.assert_fixed("A-1-12A", [True, True, True, False, False, False])

    def test_unknown_position_and_multichain_without_chain_fail(self):
        with self.assertRaisesRegex(ValueError, "matched no residue"):
            parse_fixed("A12C", self.structure)
        with self.assertRaisesRegex(ValueError, "multiple chains"):
            parse_fixed("12A", self.structure)
        with self.assertRaisesRegex(ValueError, "matched no residue"):
            parse_fixed("A12B-12A", self.structure)
        self.structure.native = "GAXYTV"
        with self.assertRaisesRegex(ValueError, "non-standard residues"):
            parse_fixed("A12A", self.structure)

    def test_manually_constructed_legacy_structure(self):
        st = self.structure
        legacy = Structure(st.name, st.coords, st.atom_mask, st.chain, st.residue_index,
                           st.native, st.labels, st.chain_names)
        np.testing.assert_array_equal(parse_fixed("A12", legacy), [False, True, True, True, False, False])

    def assert_reference_model_keeps_selected_native_residues(self, device):
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            model = HyInv.from_pretrained(device=device)
            sequences = design(model, self.structure, n=2, seed=7, fixed="A12.,A12A,B12A")
        finally:
            torch.set_num_threads(old_threads)
        self.assertEqual(len(sequences), 2)
        for sequence in sequences:
            chain_a, chain_b = sequence.split("/")
            self.assertEqual(chain_a[1:3], "AS")
            self.assertEqual(chain_b, "V")

    def test_reference_model_keeps_selected_native_residues(self):
        self.assert_reference_model_keeps_selected_native_residues("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "test requires CUDA")
    def test_reference_model_cuda_keeps_selected_native_residues(self):
        self.assert_reference_model_keeps_selected_native_residues("cuda")

    def test_command_line_uses_the_same_fixed_positions(self):
        with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as output:
            self.addCleanup(Path(output.name).unlink, missing_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        subprocess.run(
            [sys.executable, "-m", "hyinv", "design", "--pdb", self.pdb_path, "--device", "cpu",
             "--n", "1", "--fixed", "A12.,A12A,B12A", "--out", output.name],
            cwd=Path(__file__).resolve().parent.parent, env=env, check=True, capture_output=True, text=True,
        )
        lines = Path(output.name).read_text().splitlines()
        self.assertEqual(len(lines), 2)
        chain_a, chain_b = lines[1].split("/")
        self.assertEqual(chain_a[1:3], "AS")
        self.assertEqual(chain_b, "V")


if __name__ == "__main__":
    unittest.main()
