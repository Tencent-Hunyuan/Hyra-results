"""Tiny certificate fixtures from a direct, recomputing reading of the spec.

Independent of native incremental state/counters. This is test code only and
intentionally materializes tiny trees, unlike the streaming production checker.
"""
import copy
from dataclasses import dataclass
import json
import struct

from hyqec_check.algebra import Code, pauli, pauli_string
from hyqec_check.certificate import MAGIC, group_hash
from hyqec_check.trace import concatenate, entry, fingerprint
import oracle


def varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


@dataclass
class Record:
    tag: int
    fields: list
    formats: str

    def encode(self):
        result = bytes([self.tag])
        for value, fmt in zip(self.fields, self.formats):
            result += varint(value) if fmt == "U" else struct.pack("<Q", value) if fmt == "Q" else bytes([value]) if fmt == "B" else bytes(value)
        return result


def rec(tag, formats, *fields):
    return Record(tag, list(fields), formats)


def instance_text(n, rows, logicals, orbits, reps, domain):
    def sparse(row):
        support = [(q, "IXZY".index(p)) for q, p in enumerate(row) if p != "I"]
        return str(len(support)) + "".join(f" {q} {p}" for q, p in support)
    return "\n".join(["stabcert-bnb-instance 1", f"{n} {len(rows)} {len(logicals)}"]
                     + [sparse(row) for row in rows + logicals]
                     + [" ".join(map(str, orbits)), str(len(reps)) + " " + " ".join(map(str, reps)), "domain " + domain, ""])


class Fixture:
    def __init__(self, n, rows, cap, split, domain="P", permutations=(), full=False,
                 logicals=None, reverse_orbits=False, allow_logical=False, orbit_order=None):
        import hashlib
        self.code = Code.from_strings(n, rows)
        self.group = oracle.closure(n, rows) if n <= 10 else None
        self.rows = rows
        self.sparse = [[(q, "IXZY".index(p)) for q, p in enumerate(row) if p != "I"] for row in rows]
        self.cap, self.split, self.domain = cap, split, domain
        self.allow_logical = allow_logical
        self.tasks = []
        self.segments = []
        orbits = list(self.code.orbits(list(permutations)))
        if reverse_orbits:
            orbits.reverse()
        if orbit_order is not None:
            orbits = [orbits[i] for i in orbit_order]
        reps = [o[0] for o in orbits]
        indices = [next(i for i, o in enumerate(orbits) if q in o) for q in range(n)]
        if logicals is None:
            logicals = [pauli_string(v, n) for v in self.code.logicals]
        text = instance_text(n, rows, logicals, indices, reps, domain)
        self.header = dict(format="stabcert-bnb-coverage/1", search="irreducible-pauli-dfs/1",
                           trace="poly64-state-rules/1", detail="full" if full else "frontier",
                           instance_text=text, instance_sha256=hashlib.sha256(text.encode()).hexdigest(),
                           group_sha256=group_hash(self.code), n=n, m=len(rows), nl=len(logicals), k=self.code.k,
                           cap=cap, domain=domain, split=split, amax=96, permutations=list(permutations),
                           provenance={"source_sha256": "0" * 64, "binary_sha256": "1" * 64})
        top = []
        top_summary = (0, 0)
        ordinal = 0
        if cap:
            for oi, representative in enumerate(reps):
                for label in (1, 2, 3) if domain == "P" else (1,) if domain == "X" else (2,):
                    lab = tuple(label if q == representative else 0 if indices[q] < oi else
                                4 if domain == "P" else 5 if domain == "X" else 6 for q in range(n))
                    summary = self.walk(lab, top, 0, ordinal, ordinal, (), True)
                    top_summary = concatenate(top_summary, summary)
                    ordinal += 1
        self.top = top
        self.top_end = rec(0x1f, "UUUQ", len(top), len(self.tasks), *top_summary)
        self.task_records = []
        results = []
        for i, (root, path, lab) in enumerate(self.tasks):
            events = []
            summary = self.walk(lab, events, 0, 0, root, (), False)
            results.append(summary)
            records = [rec(0x21, "U", i)] + events + [rec(0x22, "U", len(events))] if full else []
            records.append(rec(0x20, "UUURUQ", i, root, len(path), path, *summary))
            self.task_records.append(records)
        total = (0, 0)
        for kind, value in self.segments:
            total = concatenate(total, results[value] if kind == "task" else value)
        task_nodes = sum(result[0] for result in results)
        assert total[0] == top_summary[0] + task_nodes
        self.footer = rec(0x7f, "UUUUQ", len(self.tasks), top_summary[0], task_nodes, *total)

    def derive(self, lab):
        value = "".join("IXZY"[a] if a < 4 else "I" for a in lab)
        active = [i for i, row in enumerate(self.rows) if not oracle.commutes(value, row)]
        sets = {i: {p: (q, tuple(x for x in (1, 2, 3) if x != g and (lab[q] == 4 or x == lab[q] - 4)))
                    for p, (q, g) in enumerate(self.sparse[i])
                    if lab[q] == 4 or (lab[q] > 4 and lab[q] - 4 != g)} for i in active}
        dead = next((i for i in active if not sets[i]), None)
        return value, active, sets, dead

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
        selected = []
        generator = 65535
        if not active:
            nontrivial = (value not in self.group if self.group is not None else
                          self.code.is_logical(pauli(value, self.code.n)))
            if nontrivial and not self.allow_logical:
                raise ValueError("fixture contains a logical witness")
            rule = 0
        elif weight + 1 > self.cap:
            rule = 1
        else:
            if weight + len(active) > self.cap:
                order = sorted(active[:96], key=lambda i: (len(sets[i]), i))
                used = set()
                for pos, g in enumerate(order):
                    if weight + len(selected) + len(order) - pos <= self.cap:
                        break
                    qs = {q for q, _ in sets[g].values()}
                    if used & qs:
                        continue
                    selected.append(g)
                    used |= qs
                    if weight + len(selected) > self.cap:
                        break
            if weight + len(selected) > self.cap:
                rule = 2
            else:
                rule = 3
                generator = min(active, key=lambda g: (sum(len(labels) for _, labels in sets[g].values()), g))
        event = len(records)
        fields = [parent, edge, rule, fingerprint(lab), weight, len(active)]
        formats = "UUBQUU"
        if rule == 2:
            fields += [len(selected)] + selected
            formats += "U" * (1 + len(selected))
        if rule == 3:
            fields.append(generator)
            formats += "U"
        records.append(Record(0x10, fields, formats))
        summary = entry(lab, weight, len(active), rule, generator, len(selected) if rule == 2 else 0)
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

    def encode(self):
        header = json.dumps(self.header, separators=(",", ":")).encode()
        records = self.top + [self.top_end] + [r for task in self.task_records for r in task] + [self.footer]
        return MAGIC + struct.pack("<Q", len(header)) + header + b"".join(r.encode() for r in records)


def block_product_fixture(blocks=350):
    """Closed-form split1 fixture: direct product of [[4,2,2]] blocks.

    More than 4096 tasks exercises streaming replay batch boundaries without
    generating a large DFS tree. Its distance is the minimum block distance,2.
    """
    import hashlib
    from hyqec_check.trace import BASE, MASK, mix64, packed
    n=4*blocks
    generators=[]
    logicals=[]
    for b in range(blocks):
        q=4*b
        generators += ['4'+''.join(f' {q+i} {p}' for i in range(4)) for p in (1,2)]
        logicals += [f'2 {q} {p} {q+j} {p}' for p in (1,2) for j in (1,2)]
    text='\n'.join(['stabcert-bnb-instance 1',f'{n} {2*blocks} {4*blocks}']+generators+logicals+
                   [' '.join(map(str,range(n))),str(n)+' '+' '.join(map(str,range(n))), 'domain P',''])
    digest=hashlib.sha256(f'stabcert-group/1 n={n} rank={2*blocks}\n'.encode())
    for offset in (0,n):
        for b in range(blocks):
            digest.update((15<<(offset+4*b)).to_bytes((2*n+7)//8,'little'))
    header=dict(format='stabcert-bnb-coverage/1',search='irreducible-pauli-dfs/1',trace='poly64-state-rules/1',
                detail='frontier',instance_text=text,instance_sha256=hashlib.sha256(text.encode()).hexdigest(),
                group_sha256=digest.hexdigest(),n=n,m=2*blocks,nl=4*blocks,k=2*blocks,cap=1,domain='P',split=1,
                amax=96,permutations=[],provenance={'source_sha256':'0'*64,'binary_sha256':'1'*64})
    initial=0
    for q in range(n):
        initial ^= mix64(8*q+5)
    top=[]; results=[]; total=(0,0)
    for q in range(n):
        for label in (1,2,3):
            task=3*q+label-1
            fp=initial^mix64(8*q+5)^mix64(8*q+label+1)
            top.append(rec(0x13,'UUUQ',0,task,task,fp))
            summary=(0,0) if q%4==3 else (1,(fp*BASE+packed(1,2 if label==3 else 1,1))&MASK)
            results.append(rec(0x20,'UUURUQ',task,task,0,(),*summary))
            total=concatenate(total,summary)
        initial ^= mix64(8*q+5)^mix64(8*q+1)
    records=top+[rec(0x1f,'UUUQ',3*n,3*n,0,0)]+results+[rec(0x7f,'UUUUQ',3*n,0,total[0],*total)]
    raw_header=json.dumps(header,separators=(',',':')).encode()
    raw=MAGIC+struct.pack('<Q',len(raw_header))+raw_header+b''.join(r.encode() for r in records)
    return raw,total
