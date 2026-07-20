#!/usr/bin/env python3
"""Parallel note-level local search over a multi-instrument arrangement, scored by the
exact scorer (orchestrate.score_solution). Started from a set of strong baseline
arrangements (seed1..seed4, kept diverse). The carrier (role=melody) part is NEVER
touched, so the hard 57-onset melody constraint can never break.

The GLOBAL best is only ever advanced by a strictly-improving, re-verified candidate, so
the written solution.json can never regress below the best starting arrangement. A fraction
of the workers use simulated-annealing acceptance so they can climb OUT of the tight local
optimum a pure greedy climber settles into; each annealer's own best is returned and
re-scored before it can affect the global best.
"""
import copy
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import orchestrate as o  # noqa: E402

MEL = o.load_melody(os.path.join(HERE, "melody.json"))

SCALE = {0, 2, 4, 5, 7, 9, 11}          # C major diatonic pitch classes
PENT = {0, 2, 4, 7, 9}                   # gong pentatonic
PMIN, PMAX = 12, 108

PROG_POOL = {
    'wind':   [73, 72, 68, 71, 74, 75],
    'string': [48, 49, 40, 41, 42, 44, 45],
    'brass':  [60, 56, 57, 61],
    'pluck':  [46, 24, 25, 105, 107, 15],
    'keys':   [0, 1, 4, 6, 11],
    'voice':  [52, 53, 54],
    'world':  [104, 105, 106, 107, 108, 110, 111],
}
ALL_PROGS = [p for v in PROG_POOL.values() for p in v]


def score_sol(sol):
    try:
        return o.score_solution(sol, MEL).score
    except Exception:
        return -1.0


def carrier_indices(sol):
    return [i for i, p in enumerate(sol['parts']) if p.get('role') == 'melody']


def snap_scale(pitch, pcset=SCALE):
    for d in [0, -1, 1, -2, 2]:
        q = pitch + d
        if PMIN <= q <= PMAX and (q % 12) in pcset:
            return q
    return max(PMIN, min(PMAX, pitch))


def valid_part(notes):
    ev = sorted(notes, key=lambda x: x[0])
    prev_end = -1
    for on, dur, pit in ev:
        if dur < 1 or on < 0:
            return False
        if on < prev_end:
            return False
        if isinstance(pit, list):
            if not pit:
                return False
            for q in pit:
                if not (PMIN <= q <= PMAX):
                    return False
        else:
            if not (PMIN <= pit <= PMAX):
                return False
        prev_end = on + dur
    return True


# ── note-level mutation operators (operate on ONE non-carrier part in place) ──────────
def _pick_note(rnd, notes):
    if not notes:
        return None
    return rnd.randrange(len(notes))


def op_pitch_shift(rnd, notes):
    j = _pick_note(rnd, notes)
    if j is None:
        return
    on, dur, pit = notes[j]
    delta = rnd.choice([-12, -7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 12])
    if isinstance(pit, list):
        k = rnd.randrange(len(pit))
        np_ = pit[k] + delta
        if PMIN <= np_ <= PMAX:
            newp = list(pit)
            newp[k] = np_
            notes[j] = [on, dur, sorted(newp)]
    else:
        np_ = pit + delta
        if PMIN <= np_ <= PMAX:
            notes[j] = [on, dur, np_]


def op_pitch_snap(rnd, notes):
    j = _pick_note(rnd, notes)
    if j is None:
        return
    on, dur, pit = notes[j]
    pcset = SCALE if rnd.random() < 0.7 else PENT
    if isinstance(pit, list):
        k = rnd.randrange(len(pit))
        newp = list(pit)
        newp[k] = snap_scale(pit[k] + rnd.choice([-3, -2, -1, 1, 2, 3]), pcset)
        notes[j] = [on, dur, sorted(set(newp)) if len(set(newp)) > 1 else [newp[0]]]
    else:
        notes[j] = [on, dur, snap_scale(pit + rnd.choice([-4, -3, -2, -1, 1, 2, 3, 4]), pcset)]


def op_split(rnd, notes):
    cand = [j for j, (on, dur, pit) in enumerate(notes) if dur >= 4 and on % 2 == 0]
    if not cand:
        return
    j = rnd.choice(cand)
    on, dur, pit = notes[j]
    half = (dur // 4) * 2
    if half < 2:
        half = 2
    if half >= dur:
        return
    if isinstance(pit, list):
        base = pit
        move = [snap_scale(q + rnd.choice([-2, 2]), SCALE) for q in pit]
        second = sorted(set(move))
        notes[j:j + 1] = [[on, half, base],
                          [on + half, dur - half, second if len(second) > 1 else (second[0] if second else base)]]
    else:
        move = snap_scale(pit + rnd.choice([-2, -1, 1, 2]), SCALE)
        notes[j:j + 1] = [[on, half, pit], [on + half, dur - half, move]]


def _chord_tones(pit):
    """Return a list of single pitches to arpeggiate over from a note's pitch value."""
    if isinstance(pit, list):
        return sorted(set(pit))
    return [pit]


def op_arpeggiate(rnd, notes):
    """Turn a long held accompaniment note/chord into a run of eighth notes stepping
    through its chord tones (and octave neighbours). Directly raises off-beat eighth-note
    motion (vitality's offbeat_rate + complementarity) while staying calm (eighth grid)."""
    cand = [j for j, (on, dur, pit) in enumerate(notes) if dur >= 6 and on % 2 == 0]
    if not cand:
        return
    j = rnd.choice(cand)
    on, dur, pit = notes[j]
    tones = _chord_tones(pit)
    if not tones:
        return
    # build extra octave/inversion tones so the arpeggio has contour
    pool = list(tones)
    for t in list(tones):
        for d in (12, -12, 7):
            q = t + d
            if PMIN <= q <= PMAX and q not in pool:
                pool.append(q)
    pool.sort()
    n_eighth = dur // 2                      # number of eighth cells
    if n_eighth < 2:
        return
    # pattern: up-run or up-down through the pool, starting near the low tone
    order = pool if rnd.random() < 0.5 else pool[::-1]
    seq = []
    t = on
    remaining = dur
    for k in range(n_eighth):
        d = 2 if k < n_eighth - 1 else remaining
        p = order[k % len(order)]
        seq.append([t, d, p])
        t += 2
        remaining -= 2
    notes[j:j + 1] = seq


def op_fill_gap(rnd, notes):
    """Insert a chord/neighbour eighth note into a REST between two of a part's notes,
    landing on the eighth grid (raises complementarity / answered gaps)."""
    if len(notes) < 2:
        return
    j = rnd.randrange(len(notes) - 1)
    on1, d1, p1 = notes[j]
    on2, d2, p2 = notes[j + 1]
    end1 = on1 + d1
    gap = on2 - end1
    if gap < 2:
        return
    start = end1 if end1 % 2 == 0 else end1 + 1
    if start >= on2:
        return
    dur = min(2, on2 - start)
    if dur < 1:
        return
    base = p1[0] if isinstance(p1, list) else p1
    pit = snap_scale(base + rnd.choice([-2, -1, 1, 2, 0]), SCALE)
    notes.insert(j + 1, [start, dur, pit])


def op_merge(rnd, notes):
    if len(notes) < 2:
        return
    j = rnd.randrange(len(notes) - 1)
    on1, d1, p1 = notes[j]
    on2, d2, p2 = notes[j + 1]
    if on1 + d1 != on2:
        return
    notes[j:j + 2] = [[on1, d1 + d2, p1]]


def op_dur(rnd, notes):
    j = _pick_note(rnd, notes)
    if j is None:
        return
    on, dur, pit = notes[j]
    nd = dur + rnd.choice([-2, 2])
    if nd >= 1:
        notes[j] = [on, nd, pit]


def op_chord_size(rnd, notes):
    cand = [j for j, (on, dur, pit) in enumerate(notes) if isinstance(pit, list)]
    if not cand:
        return
    j = rnd.choice(cand)
    on, dur, pit = notes[j]
    pit = list(pit)
    if rnd.random() < 0.5 and len(pit) < 5:
        base = rnd.choice(pit)
        add = base + rnd.choice([-12, 12, 7, -5, 4, 3])
        if PMIN <= add <= PMAX and add not in pit:
            pit.append(add)
    elif len(pit) > 2:
        pit.pop(rnd.randrange(len(pit)))
    notes[j] = [on, dur, sorted(set(pit))]


NOTE_OPS = [
    (op_pitch_snap, 3.0), (op_pitch_shift, 2.0), (op_split, 2.5),
    (op_arpeggiate, 2.0), (op_fill_gap, 1.5),
    (op_dur, 1.0), (op_chord_size, 1.5), (op_merge, 0.6),
]
_NW = [w for _, w in NOTE_OPS]
_NF = [f for f, _ in NOTE_OPS]


def op_program(rnd, sol, non_carrier):
    pi = rnd.choice(non_carrier)
    sol['parts'][pi]['program'] = rnd.choice(ALL_PROGS)


def mutate(sol, rnd, n=1):
    g = copy.deepcopy(sol)
    carriers = set(carrier_indices(g))
    non_carrier = [i for i in range(len(g['parts'])) if i not in carriers]
    if not non_carrier:
        return g
    for _ in range(n):
        if rnd.random() < 0.05:
            op_program(rnd, g, non_carrier)
            continue
        pi = rnd.choice(non_carrier)
        notes = [list(x) for x in g['parts'][pi]['notes']]
        f = rnd.choices(_NF, weights=_NW, k=1)[0]
        f(rnd, notes)
        notes.sort(key=lambda x: x[0])
        if valid_part(notes):
            g['parts'][pi]['notes'] = notes
    return g


def worker(args):
    seed, deadline, start, start_score, anneal = args
    rnd = random.Random(seed)
    cur = copy.deepcopy(start)
    cur_score = start_score
    best, best_score = copy.deepcopy(cur), cur_score
    since = 0
    # annealing temperature (in score points); ~0 for pure greedy workers
    T0 = 0.25 if anneal else 0.0
    t_start = time.time()
    span = max(1.0, deadline - t_start)
    while time.time() < deadline:
        frac = (time.time() - t_start) / span
        T = T0 * max(0.0, 1.0 - frac)        # linear cool to 0
        r = rnd.random()
        n = 1 if r < 0.7 else (2 if r < 0.92 else 3)
        cand = mutate(cur, rnd, n)
        s = score_sol(cand)
        d = s - cur_score
        accept = False
        if d > 1e-7:
            accept = True
        elif d > -1e-7:                       # equal: lateral drift
            accept = rnd.random() < 0.25
        elif T > 1e-4 and s > 0:              # annealing: accept a worse move sometimes
            accept = rnd.random() < math.exp(d / T)
        if accept:
            cur, cur_score = cand, s
            if s > best_score + 1e-9:
                best, best_score = copy.deepcopy(cand), s
                since = 0
            else:
                since += 1
        else:
            since += 1
        if since >= 300:                      # stuck: restart from best
            cur, cur_score = copy.deepcopy(best), best_score
            since = 0
    return best_score, best


def main():
    budget = float(os.environ.get("TIME_BUDGET_SEC", "600"))
    cpus = int(os.environ.get("CPUS", os.cpu_count() or 4))
    budget = max(30.0, budget - 90.0)
    deadline = time.time() + budget

    # load all seeds, keep valid ones, pick the best as the guaranteed baseline
    seeds = []
    for name in ("seed1.json", "seed2.json", "seed3.json", "seed4.json"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            try:
                s = json.load(open(p))
                sc = score_sol(s)
                if sc > 0:
                    seeds.append((sc, name, s))
                    print(f"seed {name}: {sc:.5f}", flush=True)
            except Exception as e:
                print(f"seed {name} failed: {e}", flush=True)
    seeds.sort(key=lambda x: -x[0])
    best_sol = seeds[0][2]
    best_score = seeds[0][0]
    print(f"baseline (best seed = {seeds[0][1]}): {best_score:.5f}", flush=True)
    json.dump(best_sol, open(os.path.join(HERE, "solution.json"), "w"))

    nproc = max(1, cpus)
    ROUND_LEN = min(240.0, max(45.0, budget / 8.0))
    rnd_top = random.Random(12345)
    round_i = 0
    try:
        pool = Pool(nproc)
    except Exception as e:
        print("pool init failed:", e, flush=True)
        pool = None

    while time.time() < deadline - 5:
        round_i += 1
        rdeadline = min(deadline, time.time() + ROUND_LEN)
        args = []
        for i in range(nproc):
            # ~65% of workers climb from the global best; the rest explore from the
            # diverse seeds so the search is not trapped near one arrangement.
            if i % 3 == 0 and len(seeds) > 1:
                sc, _, s = seeds[i % len(seeds)]
                start, sstart = s, sc
            else:
                start, sstart = best_sol, best_score
            anneal = (i % 2 == 1)             # half the workers anneal to escape optima
            args.append((round_i * 100003 + i * 131 + rnd_top.randrange(1_000_000),
                         rdeadline, start, sstart, anneal))
        try:
            results = pool.map(worker, args) if pool else [worker(a) for a in args]
        except Exception as e:
            print("round failed:", e, flush=True)
            results = [worker(args[0])]
        improved = False
        for s, g in results:
            if g is None:
                continue
            sc = score_sol(g)                 # re-verify with exact scorer
            if sc > best_score + 1e-9:
                best_sol, best_score = g, sc
                improved = True
        json.dump(best_sol, open(os.path.join(HERE, "solution.json"), "w"))
        print(f"round {round_i}: best={best_score:.6f} (+{'yes' if improved else 'no'})",
              flush=True)

    if pool:
        pool.close(); pool.join()

    print(f"BEST SCORE: {best_score:.6f}", flush=True)
    json.dump(best_sol, open(os.path.join(HERE, "solution.json"), "w"))
    print("wrote solution.json", flush=True)


if __name__ == "__main__":
    main()
