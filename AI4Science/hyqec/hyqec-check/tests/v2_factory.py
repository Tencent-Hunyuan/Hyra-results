"""Test-only v2 records driven by native decisions, with independent Python DFS.

These fixtures check transport/coverage, not decider conformance. Node soundness
is checked by exhaustive completion enumeration; external spec vectors supply
independent expected decisions/counts/traces.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from cert_factory import Fixture, Record, rec
from hyqec_check.algebra import pauli
from hyqec_check.trace import entry, fingerprint, concatenate
from hyqec_check_v2.certificate import write_native_model
from hyqec_check_wide.certificate import parse_instance


class NodeDriver:
    def __init__(self, fixture, rules, cap=None):
        self.temp = tempfile.TemporaryDirectory()
        model = Path(self.temp.name) / "model"
        header = dict(fixture.header, rules=rules)
        if cap is not None:
            header["cap"] = cap
        write_native_model(model, header, parse_instance(header["instance_text"]))
        self.process = subprocess.Popen([os.environ.get("HYQEC_CHECK_V2_NODE_ENGINE", "build/v2-node-driver"), str(model)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    @classmethod
    def from_rows(cls, n, rows, rules, cap, *, mode=None):
        """Raw node-rule harness only; not a certificate/algebra acceptance path."""
        out = cls.__new__(cls)
        out.temp = tempfile.TemporaryDirectory()
        model = Path(out.temp.name) / "model"
        sparse = []
        for row in rows:
            terms = [(q, "IXZY".index(p)) for q, p in enumerate(row) if p != "I"]
            sparse.append(str(len(terms)) + "".join(f" {q} {p}" for q, p in terms))
        model.write_text("\n".join([f"{n} {len(rows)} 0 {cap} 1 P frontier {rules}"] + sparse +
                                   [" ".join(map(str, range(n))),
                                    str(n) + " " + " ".join(map(str, range(n))), ""]))
        out.process = subprocess.Popen([os.environ.get("HYQEC_CHECK_V2_NODE_ENGINE", "build/v2-node-driver"), str(model)] + ([mode] if mode else []),
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        return out

    def query(self, lab):
        self.process.stdin.write(" ".join(map(str, lab)) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"node driver exited {self.process.poll()}")
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.wait(timeout=10)
        self.temp.cleanup()


class V2Fixture(Fixture):
    def __init__(self, n, rows, cap, split, domain="P", *, rules="s0s1s2", **kwargs):
        probe = Fixture(n, rows, 0, 1, domain)
        self.driver = NodeDriver(probe, rules, cap=max(1, cap))
        try:
            super().__init__(n, rows, cap, split, domain, **kwargs)
        finally:
            self.driver.close()
            del self.driver
        self.header.update(format="stabcert-bnb-coverage/2", search="irreducible-pauli-dfs/2",
                           trace="poly64-state-rules/2", rules=rules, rules_version=1)

    def encode(self):
        return b"SBCOV2\r\n" + super().encode()[8:]

    def walk(self, lab, records, parent, edge, root, path, top):
        value, active, sets, dead = self.derive(lab)
        weight = sum(a in (1, 2, 3) for a in lab)
        if top and ((self.split == 1 and not path) or (dead is None and weight == self.split)):
            task = len(self.tasks)
            records.append(rec(0x13, "UUUQ", parent, edge, task, fingerprint(lab)))
            self.tasks.append((root, path, lab))
            self.segments.append(("task", task))
            return 0, 0
        if dead is not None:
            records.append(rec(0x11, "UUUQ", parent, edge, dead, fingerprint(lab)))
            return 0, 0
        generator = 65535
        selected = []
        if not active:
            nontrivial = (value not in self.group if self.group is not None else
                          self.code.is_logical(pauli(value, self.code.n)))
            if nontrivial and not self.allow_logical:
                raise ValueError("fixture contains a logical witness")
            rule = 0
        else:
            decision = self.driver.query(lab)
            if "error" in decision:
                raise RuntimeError(decision["error"])
            rule, generator, selected = decision["rule"], decision["generator"], decision["packing"]
        event = len(records)
        fields = [parent, edge, rule, fingerprint(lab), weight, len(active)]
        formats = "UUBQUU"
        if rule in (2, 5, 6, 7):
            fields += [len(selected)] + selected
            formats += "U" * (1 + len(selected))
        if rule == 3:
            fields.append(generator)
            formats += "U"
        records.append(Record(0x10, fields, formats))
        summary = entry(lab, weight, len(active), rule, generator, len(selected) if rule in (2, 5, 6, 7) else 0)
        if top:
            self.segments.append(("entry", summary))
        if rule != 3:
            return summary
        work = list(lab)
        for position, (q, labels) in sets[generator].items():
            for li, label in enumerate(labels):
                child = work.copy()
                child[q] = label
                encoded = position * 2 + li
                summary = concatenate(summary, self.walk(tuple(child), records, event + 1, encoded,
                                                        root, path + (encoded,), top))
            work[q] = 4 + self.sparse[generator][position][1] if work[q] == 4 else 0
            suffix_dead = self.derive(work)[3]
            if suffix_dead is not None:
                records.append(rec(0x12, "UUUQ", event + 1, position, suffix_dead, fingerprint(work)))
                break
        return summary

