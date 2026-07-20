#!/usr/bin/env python3
"""pack.py - produce a SELF-CONTAINED bot source by inlining the local #include
headers (board.h, net.h, mcts.h, trainer.h) into bot.cpp, then write solution.json
as {"source": <inlined>, "language": "cpp", "files": {"weights.bin": <b64>}}.

The build step compiles only `source` with a bare `g++ -O2 -std=c++17`
(NO -I), so the submitted source must not depend on sibling headers. We expand
each local `#include "x.h"` exactly once (in dependency order), strip `#pragma
once`/include guards' re-inclusion via a seen-set, and drop the local includes;
system includes are left untouched.

Usage: pack.py [WEIGHTS_BIN] [OUT_JSON]
  WEIGHTS_BIN default: ./weights.bin   OUT_JSON default: ../solution.json
"""
import base64, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = {"board.h", "net.h", "mcts.h", "trainer.h"}


def inline(path: Path, seen: set, out: list):
    for line in path.read_text().splitlines():
        m = re.match(r'\s*#\s*include\s*"([^"]+)"', line)
        if m and m.group(1) in LOCAL:
            h = m.group(1)
            if h in seen:
                continue
            seen.add(h)
            inline(HERE / h, seen, out)
            continue
        out.append(line)


def main():
    wbin = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "weights.bin"
    out_json = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "solution.json"
    seen = set()
    lines: list[str] = []
    inline(HERE / "bot.cpp", seen, lines)
    source = "\n".join(lines) + "\n"
    sol = {"source": source, "language": "cpp"}
    if wbin.exists():
        sol["files"] = {"weights.bin": base64.b64encode(wbin.read_bytes()).decode()}
        print(f"packed weights.bin ({wbin.stat().st_size} bytes)")
    else:
        print("WARNING: no weights.bin - bot will use the warm-prior fallback")
    out_json.write_text(json.dumps(sol))
    print(f"wrote {out_json} ({out_json.stat().st_size} bytes); inlined headers: {sorted(seen)}")


if __name__ == "__main__":
    main()
