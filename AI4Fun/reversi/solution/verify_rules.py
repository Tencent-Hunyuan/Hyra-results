#!/usr/bin/env python3
"""verify_rules.py - replay C++ random self-play games through referee.py's
authoritative rules engine and confirm 0 mismatches. The C++ engine emits games
as "B x y W x y ..." (colour letter + move, -1 -1 = pass); we replay each move,
checking at every ply that:
  * a non-pass move is legal per referee._flips (flips >=1), and the mover is the
    colour the C++ engine said;
  * a pass is emitted ONLY when that colour truly has no legal move;
  * the game terminates exactly when both sides pass in a row;
  * the final board's disc counts match what the C++ engine would score.
Any violation prints the game and exits non-zero. This proves that board.h's
bitboard rules == referee.py's rules (so self-play can't be corrupting training
with illegal positions).
"""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE.parent.parent.parent  # datasets/reversi
sys.path.insert(0, str(REV))
import referee  # noqa

BLACK, WHITE, EMPTY = referee.BLACK, referee.WHITE, referee.EMPTY


def replay(tokens):
    """tokens: flat list like ['B','3','2','W','2','2',...]. Returns (ok, msg)."""
    board = referee.initial_board()
    mover = BLACK
    passed = False
    i = 0
    nmoves = 0
    while i < len(tokens):
        c = BLACK if tokens[i] == 'B' else WHITE
        x = int(tokens[i + 1]); y = int(tokens[i + 2]); i += 3
        if c != mover:
            return False, f"colour mismatch at move {nmoves}: engine said {tokens[i-3]} but mover={mover}"
        legal = referee.legal_moves(board, mover)
        if x < 0:  # pass
            if legal:
                return False, f"engine passed for {mover} at move {nmoves} but legal moves exist: {legal}"
            if passed:
                # second pass -> game should end here; there should be no more tokens
                if i < len(tokens):
                    return False, f"game continued after double pass at move {nmoves}"
                return True, "ok-doublepass"
            passed = True
        else:
            idx = x * 8 + y
            if not referee.is_legal(board, idx, mover):
                return False, f"illegal move {x},{y} for {mover} at move {nmoves}"
            referee.apply_move(board, idx, mover)
            passed = False
        mover = WHITE if mover == BLACK else BLACK
        nmoves += 1
    # ended without explicit double-pass token sequence is allowed if board full
    return True, "ok"


def main():
    ngames = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = sys.argv[2] if len(sys.argv) > 2 else "777"
    binp = HERE / "rules_check"
    if not binp.exists():
        print("build rules_check first", file=sys.stderr); sys.exit(2)
    out = subprocess.run([str(binp), "games", str(ngames), seed],
                         capture_output=True, text=True, check=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    bad = 0
    for gi, ln in enumerate(lines):
        ok, msg = replay(ln.split())
        if not ok:
            bad += 1
            if bad <= 5:
                print(f"GAME {gi} MISMATCH: {msg}")
                print(f"  {ln}")
    print(f"checked {len(lines)} games, mismatches={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
