from pathlib import Path
import json
import hashlib
import subprocess


def write_instance(code, path, domain="P", legacy=False):
    def row(value):
        pairs = [(q, ((value >> q) & 1) | (((value >> (code.n + q)) & 1) << 1)) for q in range(code.n)]
        pairs = [(q, label) for q, label in pairs if label]
        return str(len(pairs)) + "".join(f" {q} {label}" for q, label in pairs)
    lines = [] if legacy else ["stabcert-bnb-instance 1"]
    lines += [f"{code.n} {len(code.generators)} {len(code.logicals)}"]
    lines += [row(g) for g in code.generators] + [row(l) for l in code.logicals]
    lines += [" ".join(map(str, range(code.n))), str(code.n) + " " + " ".join(map(str, range(code.n)))]
    if not legacy:
        lines += [f"domain {domain}"]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def run(binary, instance, cap, *args, check=True):
    p = subprocess.run([str(binary), str(instance), "--cap", str(cap), "--threads", "2", "--quiet", "--instance-sha256", hashlib.sha256(Path(instance).read_bytes()).hexdigest(), "--binary-sha256", hashlib.sha256(Path(binary).read_bytes()).hexdigest(), *map(str, args)], capture_output=True, text=True, timeout=120)
    if check:
        assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
        records = [json.loads(line) for line in p.stdout.splitlines()]
        done = [r for r in records if r["event"] == "done"]
        assert len(done) == 1
        assert done[0]["replay_fail"] == 0
        assert done[0]["nodes"] == done[0]["gen_nodes"] + done[0]["task_nodes"]
        return done[0]
    return p


def witness(code, done):
    value = 0
    for q, label in done["witness"]:
        if label & 1:
            value |= 1 << q
        if label & 2:
            value |= 1 << (code.n + q)
    assert code.check_witness(value) == done["found"]
    return value


def python_nodes(code, cap, domain=0):
    """Tiny immutable reference tree: direct candidate sets, no native state layout.

    Only for no-find runs. The assertions ensure the supplied cap is below d.
    """
    if cap == 0:
        return 0
    gens = [[((g >> q) & 1) | (((g >> (code.n + q)) & 1) << 1) for q in range(code.n)] for g in code.generators]
    supports = [[q for q, l in enumerate(g) if l] for g in gens]
    def candidate(l, gl):
        return l == 4 or l > 4 and l - 4 != gl
    def describe(lab):
        active = [i for i, g in enumerate(gens) if sum(1 for q in supports[i] if 1 <= lab[q] <= 3 and lab[q] != g[q]) % 2]
        cands = {i: [q for q in supports[i] if candidate(lab[q], gens[i][q])] for i in active}
        return active, cands
    def visit(lab):
        active, cands = describe(lab)
        if any(not cands[i] for i in active):
            return 0
        if not active:
            v = sum(((l & 1) << q) | (((l >> 1) & 1) << (code.n + q)) for q, l in enumerate(lab) if 1 <= l <= 3)
            assert not code.is_logical(v), "python_nodes requires a no-find bound"
            return 1
        w = sum(1 <= l <= 3 for l in lab)
        if w + 1 > cap:
            return 1
        taken, bound = set(), 0
        for i in sorted(active[:96], key=lambda i: (len(cands[i]), i)):
            if taken.isdisjoint(cands[i]):
                bound += 1
                taken.update(cands[i])
        if w + bound > cap:
            return 1
        i = min(active, key=lambda i: (sum(2 if lab[q] == 4 else 1 for q in cands[i]), i))
        current, nodes = lab[:], 1
        for q in cands[i]:
            options = [l for l in (1, 2, 3) if l != gens[i][q]] if current[q] == 4 else [current[q] - 4]
            for l in options:
                child = current[:]
                child[q] = l
                nodes += visit(child)
            current[q] = 4 + gens[i][q] if current[q] == 4 else 0
            a, c = describe(current)
            if any(not c[g] for g in a):
                break
        return nodes
    total = 0
    for rep in range(code.n):
        for label in (1, 2, 3) if not domain else (domain,):
            lab = [0] * rep + [4 + domain] * (code.n - rep)
            lab[rep] = label
            total += visit(lab)
    return total


def write_ris(code, path, domain="P"):
    import struct
    from hyqec_verify.core import symplectic
    from hyqec_verify.engines.ris import _side_basis
    basis = code.normalizer_basis() if domain == "P" else _side_basis(code, domain)
    n, nl = code.n, len(code.logicals)
    nq, nsw = (n+63)//64, (nl+63)//64
    mask = (1 << n)-1
    flags = bytes(int(any((v >> c) & 1 for v in basis)) for c in range(2*n))
    data = b"PAULIRS2" + struct.pack("<4i", n, nq, len(basis), nl) + flags
    for v in basis:
        syn = sum(symplectic(v, l, n) << i for i, l in enumerate(code.logicals))
        data += (v & mask).to_bytes(8*nq,"little") + (v >> n).to_bytes(8*nq,"little") + syn.to_bytes(8*nsw,"little")
    Path(path).write_bytes(data)
    return Path(path)
