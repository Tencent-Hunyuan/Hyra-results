"""Producer-side reference code for coverage certificate version 2 (docs/BNB_CERT_SPEC_v2.md). Not a checker.

* decode / encode: the v1/v2 record stream, structure only;
* consistency: counts, trace composition, TASK_END/TASK_RESULT agreement and |PK| = h_field per rule, computed from
  the records alone (a self-consistency test of the exporter, not a re-execution of the search);
* gzip_v2 / transport_ok: the deterministic one-member gzip and a strict transport test (spec section 14);
* Inst / packing_select: a straightforward re-implementation of PACKING_SELECT (spec section 6.1) written from the
  specification text, compared with the prover's packing at every dumped prune node (cov2_pk_dump.cpp).
"""
import gzip
import io
import json
import struct
import zlib
from pathlib import Path

# ---------------------------------------------------------------- stream structure

class Stream:
    def __init__(self, raw):
        self.f = io.BytesIO(raw)

    def byte(self):
        b = self.f.read(1)
        if not b:
            raise ValueError("truncated")
        return b[0]

    def fixed(self):
        b = self.f.read(8)
        if len(b) != 8:
            raise ValueError("truncated u64")
        return struct.unpack("<Q", b)[0]

    def u(self):
        result = shift = 0
        for i in range(10):
            b = self.byte()
            result |= (b & 127) << shift
            if not b & 128:
                if i and not b:
                    raise ValueError("nonminimal LEB128")
                if i == 9 and b > 1:
                    raise ValueError("LEB128 overflow")
                return result
            shift += 7
        raise ValueError("LEB128 longer than 10 bytes")


def decode(raw):
    """(magic version, header, records). Records are tuples; ENTER = (0x10, parent, edge, rule, fp, w, nA, hint)."""
    s = Stream(raw)
    magic = s.f.read(8)
    version = {b"SBCOV1\r\n": 1, b"SBCOV2\r\n": 2}[magic]
    size = s.fixed()
    header = json.loads(s.f.read(size), object_pairs_hook=_nodup)
    v2rules = version == 2 and header["rules"] != "greedy"
    recs = []
    while True:
        tag = s.byte()
        if tag == 0x10:
            parent, edge, rule, fp, w, na = s.u(), s.u(), s.byte(), s.fixed(), s.u(), s.u()
            if rule == 2 or (v2rules and rule in (5, 6, 7)):
                hint = [s.u() for _ in range(s.u())]
            elif rule == 3:
                hint = [s.u()]
            elif rule in (0, 1):
                hint = []
            else:
                raise ValueError(f"rule {rule} not permitted")
            recs.append((tag, parent, edge, rule, fp, w, na, hint))
        elif tag in (0x11, 0x12, 0x13):
            recs.append((tag, s.u(), s.u(), s.u(), s.fixed()))
        elif tag == 0x1f:
            recs.append((tag, s.u(), s.u(), s.u(), s.fixed()))
        elif tag == 0x20:
            task, root, length = s.u(), s.u(), s.u()
            path = s.f.read(length)
            recs.append((tag, task, root, path, s.u(), s.fixed()))
        elif tag in (0x21, 0x22):
            recs.append((tag, s.u()))
        elif tag == 0x7f:
            recs.append((tag, s.u(), s.u(), s.u(), s.u(), s.fixed()))
            if s.f.read(1):
                raise ValueError("bytes after COMPLETE")
            return version, header, recs
        else:
            raise ValueError(f"unknown tag {tag:#x}")


def _nodup(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate key {k}")
        d[k] = v
    return d


def encode(version, header, recs):
    """Inverse of decode (used to build deliberately-invalid artifacts)."""
    out = bytearray({1: b"SBCOV1\r\n", 2: b"SBCOV2\r\n"}[version])
    hb = json.dumps(header, separators=(",", ":")).encode()
    out += struct.pack("<Q", len(hb)) + hb

    def u(v):
        while True:
            b = v & 127
            v >>= 7
            out.append(b | (128 if v else 0))
            if not v:
                return

    def fixed(v):
        out.extend(struct.pack("<Q", v))
    for r in recs:
        tag = r[0]
        out.append(tag)
        if tag == 0x10:
            _, parent, edge, rule, fp, w, na, hint = r
            u(parent); u(edge); out.append(rule); fixed(fp); u(w); u(na)
            if rule == 3:
                u(hint[0])
            elif rule not in (0, 1):
                u(len(hint))
                for g in hint:
                    u(g)
        elif tag in (0x11, 0x12, 0x13, 0x1f):
            u(r[1]); u(r[2]); u(r[3]); fixed(r[4])
        elif tag == 0x20:
            u(r[1]); u(r[2]); u(len(r[3])); out.extend(r[3]); u(r[4]); fixed(r[5])
        elif tag in (0x21, 0x22):
            u(r[1])
        elif tag == 0x7f:
            u(r[1]); u(r[2]); u(r[3]); u(r[4]); fixed(r[5])
    return bytes(out)


def header_bytes(raw):
    size = struct.unpack("<Q", raw[8:16])[0]
    return raw[:16 + size], raw[16 + size:]


# ---------------------------------------------------------------- gzip transport

def gzip_v2(raw: bytes) -> bytes:
    """Exactly one member, mtime 0, no file name (the package's _compress_coverage settings)."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=6) as g:
        g.write(raw)
    return buf.getvalue()


def transport_ok(gz: bytes):
    """Strict v2 transport (spec 14): one member, nothing after it, CRC-32 and ISIZE correct."""
    if gz[:2] != b"\x1f\x8b" or gz[2] != 8:
        return False, "not gzip/deflate"
    flags = gz[3]
    if flags & 0x08:
        return False, "has original file name"
    if struct.unpack("<I", gz[4:8])[0] != 0:
        return False, "mtime nonzero"
    d = zlib.decompressobj(wbits=31)
    try:
        data = d.decompress(gz) + d.flush()
    except zlib.error as e:
        return False, f"zlib: {e}"
    if not d.eof:
        return False, "truncated member"
    if d.unused_data:
        return False, f"{len(d.unused_data)} bytes after the member"
    crc, isize = struct.unpack("<II", gz[-8:])
    if crc != zlib.crc32(data) or isize != len(data) & 0xffffffff:
        return False, "trailer mismatch"
    return True, data


# ---------------------------------------------------------------- spec 6.1 re-implementation

class Inst:
    """Parsed instance text (spec 2): generator rows as lists of (qubit, label)."""

    def __init__(self, text):
        t = text.split()
        pos = 2 if t[0] == "stabcert-bnb-instance" else 0
        n, m, nl = int(t[pos]), int(t[pos + 1]), int(t[pos + 2])
        pos += 3
        rows = []
        for _ in range(m + nl):
            r = int(t[pos]); pos += 1
            rows.append([(int(t[pos + 2 * i]), int(t[pos + 2 * i + 1])) for i in range(r)])
            pos += 2 * r
        self.n, self.m, self.gens = n, m, rows[:m]


def node_quantities(inst, lab):
    """s(g), C(g) (candidate qubit sets in support order) for a state lab[0..n)."""
    s, C = [], []
    for g in inst.gens:
        par = 0
        cs = []
        for q, a in g:
            x = lab[q]
            if 1 <= x <= 3 and x != a:
                par ^= 1
            if x == 4 or (x > 4 and x - 4 != a):
                cs.append(q)
        s.append(par)
        C.append(cs)
    return s, C


PATHS = {}   # which branch of 6.1 produced each PK: a census, so tests can require every branch to be exercised


def _path(tag):
    PATHS[tag] = PATHS.get(tag, 0) + 1


def packing_select(inst, lab, cap, rules):
    """(np, PK) per BNB_CERT_SPEC_v2 6.1, or None when 5.2 does not run it (R < r)."""
    s, C = node_quantities(inst, lab)
    w = sum(1 for x in lab if 1 <= x <= 3)
    r = cap - w
    A = [g for g in range(inst.m) if s[g]]
    R = len(A)
    if R < r:
        return None
    # (i) greedy packing over the first min(96, R) active generators (index order), ordered by (cnt, g)
    L = sorted(A[:96], key=lambda g: (len(C[g]), g))
    used, PK = set(), []
    for g in L:
        if len(PK) == r + 1:
            break
        if used & set(C[g]):
            continue
        PK.append(g)
        used |= set(C[g])
    np_ = len(PK)
    if R > 96:
        _path("R>96")
    if np_ >= r or R > 64:
        _path("greedy" if np_ >= r else "greedy-R>64")
        return np_, PK
    return mis_threshold(A, C, r, rules, np_, PK)


def mis_threshold(V, C, r, rules, lb, PK):
    R = len(V)
    adj = [set(j2 for j2 in range(R) if j2 != j and set(C[V[j]]) & set(C[V[j2]])) for j in range(R)]

    def mindeg(X):
        best = None
        for u in sorted(X):
            d = len(adj[u] & X)
            if best is None or d < best[0]:
                best = (d, u)
        return best[1]
    # (a)
    X, Sg = set(range(R)), set()
    while X:
        u = mindeg(X)
        Sg.add(u)
        X -= {u} | adj[u]
    if len(Sg) > lb:
        PK = [V[j] for j in sorted(Sg)][:min(len(Sg), r + 1)]
        if len(Sg) >= r:
            _path("mis-a" + (">r" if len(Sg) > r else ""))
            return len(Sg), PK
        lb = len(Sg)
    # (b)
    Q = []
    for j in range(R):
        for q in C[V[j]]:
            if q not in Q:
                Q.append(q)
    M = {q: {j for j in range(R) if q in C[V[j]]} for q in Q}
    U, cover = set(range(R)), 0
    while U and cover < r:
        b, qb = -1, None
        for q in Q:
            if len(M[q] & U) > b:
                b, qb = len(M[q] & U), q
        if b <= 1:
            cover += len(U)
            U = set()
            break
        U -= M[qb]
        cover += 1
    ub = cover if not U else r
    T = r - 1 if rules in ("s0s1", "s0s1s2") else r
    if ub < T or ub <= lb:
        _path("mis-b-stop")
        return lb, PK
    # (c)
    st = {"best": lb, "bs": None}

    def search(X, cur, chosen):
        if st["best"] >= ub:
            return
        if not X:
            if cur > st["best"]:
                st["best"], st["bs"] = cur, chosen
            return
        if cur + len(X) <= st["best"]:
            return
        u = mindeg(X)
        d = len(adj[u] & X)
        search(X - ({u} | adj[u]), cur + 1, chosen | {u})
        if d > 0:
            search(X - {u}, cur, chosen)
    search(set(range(R)), 0, frozenset())
    if st["bs"] is None:
        _path("mis-c-none")
        return lb, PK
    best = st["best"]
    _path("mis-c" + (">r" if best > r else ""))
    return best, [V[j] for j in sorted(st["bs"])][:min(best, r + 1)]


RULE_OF = {"packing": 2, "s0": 5, "s1": 6, "s2": 7}


def check_dump(inst, dump_path, rules):
    """Every dumped prune node: spec PK == prover PK, and the 5.2 dispatch admits the rule. Returns stats."""
    stats, bad = {}, []
    lines = Path(dump_path).read_text().splitlines() if Path(dump_path).exists() else []
    for line in lines:
        parts = line.split()
        why, cap, w, fp, labs = parts[0], int(parts[1]), int(parts[2]), parts[3], parts[4]
        pk = [int(x) for x in parts[5:]]
        lab = [int(ch) for ch in labs]
        r = cap - w
        got = packing_select(inst, lab, cap, rules)
        rule = RULE_OF[why]
        stats[rule] = stats.get(rule, 0) + 1
        if got is None:
            bad.append((line, "spec does not run the rules (R < r)"))
            continue
        np_, spk = got
        if spk != pk:
            bad.append((line, f"PK mismatch: spec {spk}"))
        if len(spk) != min(np_, r + 1):
            bad.append((line, "spec PK length"))
        ok = {2: np_ > r,
              5: np_ == r and not (rules == "s0s1s2" and np_ == r - 2),
              6: rules in ("s0s1", "s0s1s2") and np_ == r - 1 and r >= 2,
              7: rules == "s0s1s2" and np_ == r - 2}[rule]
        if not ok:
            bad.append((line, f"dispatch: np={np_} r={r} rule={rule}"))
    return stats, bad


# ---------------------------------------------------------------- record-level self-consistency (not a checker)

MOD = 1 << 64
BASE = 0x100000001b3


def entry_word(version, rules, rec):
    _, _, _, rule, fp, w, na, hint = rec
    g = hint[0] if rule == 3 else 65535
    h = len(hint) if rule == 2 or (version == 2 and rules != "greedy" and rule in (5, 6, 7)) else 0
    return fp, (w | (na << 8) | (rule << 24) | (g << 32) | (h << 48)) % MOD


class Summ:
    def __init__(self):
        self.n, self.h = 0, 0

    def entry(self, fp, word):
        self.n += 1
        self.h = ((self.h * BASE + fp) * BASE + word) % MOD

    def append(self, n, h):
        self.h = (self.h * pow(BASE, 2 * n, MOD) + h) % MOD
        self.n += n


ALLOWED = {"greedy": {0, 1, 2, 3}, "s0": {0, 1, 2, 3, 5}, "s0s1": {0, 1, 2, 3, 5, 6}, "s0s1s2": {0, 1, 2, 3, 5, 6, 7}}


def consistency(raw, done=None):
    """Structure, counts, traces and h_field/PK length of one artifact. Returns a census; raises on a problem."""
    version, hd, recs = decode(raw)
    rules = hd.get("rules", "greedy") if version == 2 else "greedy"
    if version == 2:
        assert (hd["format"], hd["search"], hd["trace"], hd["rules_version"]) == (
            "stabcert-bnb-coverage/2", "irreducible-pauli-dfs/2", "poly64-state-rules/2", 1)
        assert rules in ALLOWED
    cap = hd["cap"]
    census = {"top": {}, "task": {}}

    def check_enter(r, where):
        rule, w, hint = r[3], r[5], r[7]
        assert rule in ALLOWED[rules], f"rule {rule} under {rules}"
        census[where][rule] = census[where].get(rule, 0) + 1
        if version == 2 and rules != "greedy" and rule in (2, 5, 6, 7):
            want = {2: cap - w + 1, 5: cap - w, 6: cap - w - 1, 7: cap - w - 2}[rule]
            assert len(hint) == want, f"|PK| {len(hint)} != {want} (rule {rule})"
            assert len(set(hint)) == len(hint) and all(0 <= g < hd["m"] for g in hint)
    i = 0
    top, seg, segs, nfront, events = Summ(), Summ(), [], 0, 0
    while recs[i][0] != 0x1f:
        r = recs[i]
        events += 1
        if r[0] == 0x10:
            check_enter(r, "top")
            fp, wd = entry_word(version, rules, r)
            top.entry(fp, wd); seg.entry(fp, wd)
        elif r[0] == 0x13:
            assert r[3] == nfront, "frontier task ids"
            nfront += 1
            segs.append(seg); seg = Summ()
        i += 1
    segs.append(seg)
    _, ev, tc, gen, th = recs[i]
    assert (ev, tc, gen, th) == (events, nfront, top.n, top.h), "TOP_END"
    i += 1
    glob, tasks_sum, tid = Summ(), 0, 0
    while recs[i][0] != 0x7f:
        r = recs[i]
        if r[0] == 0x21:
            assert r[1] == tid
            t, cnt = Summ(), 0
            i += 1
            while recs[i][0] != 0x22:
                cnt += 1
                if recs[i][0] == 0x10:
                    check_enter(recs[i], "task")
                    t.entry(*entry_word(version, rules, recs[i]))
                else:
                    assert recs[i][0] in (0x11, 0x12), "no FRONTIER inside a transcript"
                i += 1
            assert recs[i][1] == cnt, "TASK_END event_count"
            i += 1
            r = recs[i]
            assert r[0] == 0x20 and (r[4], r[5]) == (t.n, t.h), "FULL transcript vs TASK_RESULT"
        assert r[0] == 0x20 and r[1] == tid, "task order"
        glob.append(segs[tid].n, segs[tid].h)
        glob.append(r[4], r[5])
        tasks_sum += r[4]
        tid += 1
        i += 1
    glob.append(segs[tid].n, segs[tid].h)
    _, tc2, gen2, tn, tot, gt = recs[i]
    assert tid == nfront == tc2 and gen2 == top.n and tn == tasks_sum and tot == gen2 + tn, "COMPLETE counts"
    assert glob.n == tot and glob.h == gt, "global trace composition"
    if done is not None:
        assert done["nodes"] == tot and done["gen_nodes"] == gen2 and done["task_nodes"] == tn
        assert int(done["global_trace"], 16) == gt
    census.update(version=version, rules=rules, nodes=tot, tasks=tid, gen=gen2, trace=f"{gt:016x}")
    return census
