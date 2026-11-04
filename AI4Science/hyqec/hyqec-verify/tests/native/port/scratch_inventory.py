#!/usr/bin/env python3
"""List every data member of the S0/S1/S2 rule code (native/bnb/s012.inc: members of struct Dfs; s2.inc: members of
struct S2Scratch) and check that each one is covered by tests/native/port/SCRATCH_AUDIT.md (owner requirement 1:
every per-node cache needs a bounded-cleanup argument). Usage: scratch_inventory.py [--list]."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BNB = ROOT / 'src/hyqec_verify/native/bnb'
AUDIT = Path(__file__).with_name('SCRATCH_AUDIT.md')
TYPES = r'(?:static\s+const\s+)?(?:u?int(?:8|16|32|64)_t|int|long long|bool|uint32_t)'
# a member declaration: exactly two-space indentation at struct scope, a type, then declarators up to ';'
DECL = re.compile(r'^  ' + TYPES + r'\s+([^;]+);')


def members(path):
    """Declarations on lines at struct scope (exactly two-space indentation; function bodies are deeper). A line may
    hold several declarations of different types separated by ';'."""
    names = []
    text = path.read_text()
    if path.name == 's2.inc':                         # only struct S2Scratch (s2_extra is a free function)
        text = text[text.index('struct S2Scratch'):text.index('inline int s2_extra')]
    for line in text.split('\n'):
        if not line.startswith('  ') or line.startswith('   '):
            continue
        for stmt in line.split('//')[0].split(';')[:-1]:
            stmt = stmt.strip()
            if '(' in stmt and '=' not in stmt.split('(')[0]:
                continue                              # a function, not a data member
            m = DECL.match('  ' + stmt + ';')
            if not m:
                continue
            for part in m.group(1).split(','):
                name = re.match(r'\s*([A-Za-z_]\w*)', part)
                if name:
                    names.append(name.group(1))
    return names


def main():
    found = {f: members(BNB / f) for f in ('s012.inc', 's2.inc')}
    text = AUDIT.read_text()
    missing = [(f, n) for f, ns in found.items() for n in ns if not re.search(r'\b' + re.escape(n) + r'\b', text)]
    if '--list' in sys.argv:
        for f, ns in found.items():
            print(f, len(ns), ' '.join(ns))
    if missing:
        print('NOT IN SCRATCH_AUDIT.md:', ' '.join(f'{f}:{n}' for f, n in missing))
        return 1
    print(f'all {sum(map(len, found.values()))} members covered')
    return 0


if __name__ == '__main__':
    sys.exit(main())
