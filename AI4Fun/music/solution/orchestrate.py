"""orchestrate - a self-contained, deterministic scorer for the multi-instrument
arrangement task.

A single module that defines what an arrangement IS and how it SCORES, so scoring is
deterministic and fully reproducible from this file alone.

THE TASK
========
Arrange the fixed gong-pentatonic melody of 望春風 / Bāng-chhun-hong (Teng Yu-hsien,
1933 - see ``melody.json``, the COMPLETE through-composed verse, 57 notes in four
distinct phrases, key C major) for a **free ensemble of 2–16 instruments**. The arranger
chooses the instrumentation (any General-MIDI programs), and writes each instrument an
independent line of timed notes, chords and rests on the melody's sixteenth-note grid. ONE
part carries the human tune; the rest accompany it - with independent, moving lines that
ANSWER the tune in its gaps, not a homorhythmic chorale struck in lockstep.

A submission is::

    {"parts": [
        {"name": "Flute",   "program": 73, "role": "melody",
         "notes": [[0, 6, 67], [6, 2, 71], ...]},          # [onset16, dur16, pitch]
        {"name": "Strings", "program": 48, "role": "harmony",
         "notes": [[0, 16, [60, 64, 67]], ...]},            # a chord = list of pitches
        {"name": "Cello",   "program": 42, "role": "bass",
         "notes": [[0, 8, 36], ...]}],
     "tempo_bpm": 88}

``program`` is a GM instrument 0..127; ``role`` ∈ {melody,harmony,bass,counter,color}
(informational - the carrier is found structurally); each note is ``[onset16, dur16,
pitch]`` with ``pitch`` a MIDI int OR a list of ints (the chord that part sounds);
gaps between notes are rests.

HARD CONSTRAINTS → an INVALID submission (a ``ValueError`` is raised, never a silent
zero):
  1. schema / 2..16 parts / GM program 0..127 / pitches in MIDI [12,108] / ``dur16``≥1 /
     notes inside the piece; exact-duplicate notes are canonicalized away.
  2. the MELODY-CARRIER gate (robust, onset-anchored): at every one of the 57 ground-
     truth melody onsets, at least one ``role:"melody"`` note must BEGIN exactly there
     with pitch ``= ground_truth + off`` where ``off`` is a whole number of octaves
     (``off % 12 == 0``, ``|off| ≤ 24``) and ``off`` is CONSTANT within each phrase (the
     octave may change only at the four phrase boundaries - so the tune can pass between
     instruments, restate it an octave higher between phrases, but its contour is never
     distorted).
     ``role:"melody"`` notes that begin BETWEEN the ground-truth onsets are ornaments
     (validated as figuration, not part of the skeleton). No carrier, a wrong-pitch
     carrier, or a mid-phrase octave jump → invalid. (Whether the tune is the TOP voice
     is a soft *scored* term, not a hard gate, so a brief countermelody above it is OK.)

THE SCORE (0..100, HIGHER is better) is the
human-aesthetic prior made checkable - a purely DETERMINISTIC blend of four axes, then
**gated** by anti-degenerate multipliers (the structural, ungameable spine)::

    base  = wH·HARMONY + wO·ORCHESTRATION + wI·INTEREST + wF·FOUNDATION   (each 0..1)
    score = 100 · base · G_usage · G_content · G_harmrhythm · G_voicing · G_register
                       · G_contrast · G_spread · G_balance · G_pace · G_continuity · floor_pen

  HARMONY is the PRIMARY axis (the user's first aim - correct harmony above all - so it
  carries the largest base weight). The four axes:
  * HARMONY       - at every melody onset the full sounding chord is a consonant, in-key,
    functional sonority that supports the tune. The recognizer rewards complete in-key
    chords - diatonic triads (I/ii/iii/IV/V/vi) AND their colour extensions (6ths, 7ths,
    add9) EQUALLY (a held colour chord is first-class, not a defect to be optimized away) -
    gives sus / incomplete chords partial credit, and PENALIZES genuine off-key writing
    (notes outside the C-major scale) and clusters; cadences at phrase ends. The chord-
    completeness reward and the off-key penalty are measured on the ACCOMPANIMENT mass (the
    pentatonic-melody-over-triad sus/add9 colouring of the FULL chord is correct).
  * ORCHESTRATION - registral spacing (overtone principle: wide low, close high; a "mud"
    penalty for close intervals in the bass), idiomatic octave doubling, each instrument
    written in its comfortable range, textural variety across the phrases, a rich PALETTE
    (many distinct GM programs across several instrument FAMILIES, layered across the four
    bass→soprano registral strata), and melody salience.
  * INTEREST      - independent moving inner voices / counter-melodies (in calm eighth-note
    motion), DEVELOPMENT of the orchestration across the four distinct phrases (texture
    varied, not one figure run throughout), a shaped density contour, an entropy that sits
    between monotony and noise, tasteful figuration (off-beat non-chord tones treated
    correctly - vitality is gated by figuration correctness, and only EIGHTH-grid motion
    earns vitality credit, not sixteenth flurries), and COMPLEMENTARY RHYTHM with the tune
    (节奏对齐 as call-and-response - the accompaniment answers the melody in its held-note
    gaps with a genuinely MOVING line, rather than striking in homorhythmic lockstep; a
    block chorale, a same-note hammer, and a frozen drone all earn only the floor - this is
    a composition task, not a chorale).
  * FOUNDATION    - a real bass register, chord-tone bass, good bass motion (leaps at chord
    changes, steps or an arpeggiated walk through INVERSIONS within a held chord). Inversions
    are REWARDED, not merely tolerated: an all-root-position bass (root under every chord) is
    mildly docked, because the only way to give it "motion" is to keep re-rooting the harmony.

  The gates each collapse to 0 (or a floor) on the degenerate they guard and sit at ~1 for
  genuine music (a flat weighted sum is gamed by farming the top term and flooring the
  rest): G_usage (declare-then-ignore, or barely-played, instruments - min-sensitive over
  per-part participation), G_content (mere melody doubling), G_harmrhythm (a one-chord drone
  at one end AND, at the other, ROOT CHURN - re-rooting the harmony on nearly every note;
  both a frozen and a frantic harmonic rhythm are docked, so the sweet spot is ~1 chord per
  couple of beats with the bass moving by inversion within it), G_voicing (in-key by PC-set
  but hollow / non-chord bass / static pedal), G_register (out-of-range writing), G_contrast
  (one texture run under every phrase, no development), G_spread (a single stratospheric note
  faking a wide registral spread), G_balance (a pile of near-duplicate doubling parts, an
  over-dense wall, or a SAME-LINE WALL - several instruments in different timbres all doubling
  one line - guards the WIDENED up-to-16-part action space so a bigger ensemble only helps
  when each part is a genuinely DISTINCT line), G_pace (a frantic sixteenth-note-heavy
  accompaniment - the song wants calm, singable eighth/quarter motion; floors at 0.55, a
  taste cost, not a zero), G_continuity (a part that enters or drops out in the MIDDLE of a
  phrase, or a whole block cutting out at once - a sudden textural hole; instruments may
  change only at phrase boundaries).

Determinism: pure Python + numpy, no RNG on the scoring path. Same submission + same
melody → identical score on any machine. The renderers (mp3 / MIDI / piano-roll PNG)
are STRICTLY fail-soft - a missing audio lib drops the artifact, never the score; and
they prefer a sampled-instrument FluidSynth render, falling back to a polished numpy
synth, so scoring still works on a bare numpy+stdlib image.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

# ── C-major diatonic harmony, encoded once. vii° is diminished; cadential sevenths
#    are added; a palette of colour chords (sus / add9 / 6ths / 7ths) is recognized so
#    the harmony can be rich, not just plain triads. ─────────────────────────────────
MAJOR_SCALE_PCS = (0, 2, 4, 5, 7, 9, 11)
SCALE_SET = frozenset(MAJOR_SCALE_PCS)
TONIC_PC = 0

DIATONIC_TRIADS = {
    0:  ("I",   (0, 4, 7), "maj"),
    2:  ("ii",  (2, 5, 9), "min"),
    4:  ("iii", (4, 7, 11), "min"),
    5:  ("IV",  (5, 9, 0), "maj"),
    7:  ("V",   (7, 11, 2), "maj"),
    9:  ("vi",  (9, 0, 4), "min"),
    11: ("vii", (11, 2, 5), "dim"),
}
DIATONIC_SEVENTHS = {
    7:  ("V7",   (7, 11, 2, 5)),
    11: ("vii7", (11, 2, 5, 9)),
    2:  ("ii7",  (2, 5, 9, 0)),
}
_ROMAN_BY_ROOT = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii"}

# The in-key sonority FRAMES a single sustained chord may occupy - every diatonic triad
# plus its in-scale colour extensions (6th / add9 / maj7|min7|dom7 / sus). ``_fits_one_chord``
# tests membership against these, so a HELD colour chord (a sustained I6 / IV add9 / V7) reads
# as ONE harmony rather than a spurious change on every reattack. Built once at import from
# the recognized templates, keeping only frames wholly inside C major (a genuine functional
# or off-key move spells a set outside every frame and still counts as a change).
def _build_diatonic_frames() -> tuple:
    frames = set()
    ext_templates = [(0, 4, 7), (0, 3, 7),            # triads
                     (0, 4, 7, 9), (0, 3, 7, 9),      # 6ths
                     (0, 2, 4, 7),                    # add9
                     (0, 4, 7, 10), (0, 4, 7, 11), (0, 3, 7, 10),  # 7ths
                     (0, 2, 7), (0, 5, 7)]            # sus2 / sus4
    for root in range(12):
        for tmpl in ext_templates:
            pcs = frozenset((root + iv) % 12 for iv in tmpl)
            if pcs.issubset(SCALE_SET):
                frames.add(pcs)
    return tuple(frames)


_DIATONIC_CHORD_FRAMES = _build_diatonic_frames()

# Chord templates (intervals above a root) recognized as REAL, intentional sonorities,
# each with a base quality score in [0,1] when it sits inside the C-major scale. A
# template matching but using a chromatic note is reduced to a mild "colour/secondary"
# score (musical, slightly out of key). Keyed by a frozenset of intervals so e.g.
# (0,2,4,7) and a reordering collapse to one entry. ``has_third`` marks a defined major/
# minor quality (vs an ambiguous sus / power chord).
_CHORD_TEMPLATES = [
    ((0, 4, 7),        "maj",   1.00, True),
    ((0, 3, 7),        "min",   1.00, True),
    ((0, 3, 6),        "dim",   0.90, True),
    ((0, 4, 8),        "aug",   0.70, True),
    ((0, 4, 7, 10),    "dom7",  0.97, True),
    ((0, 4, 7, 11),    "maj7",  0.96, True),
    ((0, 3, 7, 10),    "min7",  0.96, True),
    ((0, 3, 6, 10),    "m7b5",  0.90, True),
    ((0, 3, 6, 9),     "dim7",  0.85, True),
    ((0, 4, 7, 9),     "maj6",  0.98, True),
    ((0, 3, 7, 9),     "min6",  0.97, True),
    ((0, 2, 7),        "sus2",  0.90, False),
    ((0, 5, 7),        "sus4",  0.90, False),
    ((0, 2, 4, 7),     "add9",  0.98, True),
    ((0, 7),           "power", 0.55, False),
]
# NOTE the colour chords (6ths / 7ths / add9) are scored ON A PAR with plain triads
# (~0.96–0.98, only a hair below a bare 1.00 major/minor). This is deliberate: a gong-
# pentatonic melody sustained over a held diatonic triad naturally sounds a 6th / 9th /
# sus against it, so the FULL vertical READS as a colour chord - that floating richness is
# the DESIRED sound (the melody is not shackled to the plainest triad that contains it), not
# a defect to be optimized away. Scoring colour flush with triads removes the old gradient
# that pushed an optimizer to re-root the harmony on every melody note just to keep every
# vertical a bare triad (the "over-consonant, root-churning" attractor the user flagged).

_TERTIAN_TEMPLATES = (
    (0, 4, 7), (0, 3, 7), (0, 3, 6), (0, 4, 8),
    (0, 4, 7, 10), (0, 4, 7, 11), (0, 3, 7, 10), (0, 3, 6, 9), (0, 3, 6, 10),
)

# Root-motion quality by ascending PC interval (functional flow; descending fifths
# strongest). Same table as the sibling tasks.
_ROOT_MOTION_Q = {
    5: 1.00, 2: 0.85, 9: 0.80, 7: 0.60, 10: 0.55, 3: 0.55,
    8: 0.50, 4: 0.50, 6: 0.45, 1: 0.40, 11: 0.40, 0: 0.20,
}

# Down-weight the weak/ambiguous mediant so an optimizer cannot farm iii as free full
# credit (a real I/IV/V/ii/vi plan should win), mirroring guitar_arrange.
_FUNCTION_WEIGHT = {"iii": 0.85}


# ── instrument ranges (sounding MIDI), keyed by GM program. PLAY = absolute playable
#    extremes (outside → the register gate bites); COMF = comfortable/characteristic
#    band (outside but inside PLAY → a soft cost, like the SATB voice ranges). Standard
#    conservative pedagogical values (Adler / Piston) for the acoustic orchestra; the
#    synth/world/FX programs get sensible practical bands. ~ALL 128 GM programs are
#    covered so the register check + range-fit are correct for any instrument the
#    arranger picks (incl. the ethnic/world block, idiomatic for this folk tune); the
#    family-bucket fallback in ``inst_range`` remains a safety net. ───────────────────
_INSTRUMENT_RANGES = {
    # ── 0-7 pianos ──
    0: ((21, 108), (28, 100)),  # acoustic grand
    1: ((21, 108), (28, 100)),  # bright acoustic
    2: ((21, 108), (28, 100)),  # electric grand
    3: ((21, 108), (28, 100)),  # honky-tonk
    4: ((28, 103), (33, 96)),   # electric piano 1 (Rhodes)
    5: ((28, 103), (33, 96)),   # electric piano 2 (FM)
    6: ((36, 96), (40, 88)),    # harpsichord
    7: ((36, 96), (40, 88)),    # clavinet
    # ── 8-15 chromatic percussion ──
    8: ((48, 96), (53, 89)),    # celesta
    9: ((60, 108), (64, 103)),  # glockenspiel
    10: ((60, 96), (64, 91)),   # music box
    11: ((53, 89), (53, 84)),   # vibraphone
    12: ((45, 96), (48, 91)),   # marimba
    13: ((72, 108), (76, 103)), # xylophone
    14: ((43, 79), (48, 74)),   # tubular bells
    15: ((53, 89), (57, 84)),   # dulcimer (hammered)
    # ── 16-23 organ ──
    16: ((24, 100), (36, 91)),  # drawbar organ
    17: ((24, 100), (36, 91)),  # percussive organ
    18: ((24, 100), (36, 91)),  # rock organ
    19: ((24, 103), (36, 96)),  # church organ
    20: ((24, 100), (36, 91)),  # reed organ
    21: ((41, 96), (48, 89)),   # accordion
    22: ((41, 96), (48, 89)),   # harmonica
    23: ((41, 96), (48, 89)),   # tango accordion
    # ── 24-31 guitar ──
    24: ((40, 83), (40, 76)),   # nylon guitar
    25: ((40, 83), (40, 76)),   # steel guitar
    26: ((40, 86), (40, 79)),   # jazz electric
    27: ((40, 86), (40, 79)),   # clean electric
    28: ((40, 86), (40, 79)),   # muted electric
    29: ((40, 88), (40, 81)),   # overdriven
    30: ((40, 88), (40, 81)),   # distortion
    31: ((40, 86), (40, 79)),   # harmonics
    # ── 32-39 bass ──
    32: ((28, 60), (28, 55)),   # acoustic bass
    33: ((28, 67), (28, 60)),   # fingered electric bass
    34: ((28, 67), (28, 60)),   # picked electric bass
    35: ((28, 67), (28, 60)),   # fretless bass
    36: ((28, 67), (28, 60)),   # slap bass 1
    37: ((28, 67), (28, 60)),   # slap bass 2
    38: ((28, 67), (28, 60)),   # synth bass 1
    39: ((28, 67), (28, 60)),   # synth bass 2
    # ── 40-47 strings / orchestral ──
    40: ((55, 100), (55, 93)),  # violin
    41: ((48, 88), (48, 81)),   # viola
    42: ((36, 84), (36, 76)),   # cello
    43: ((28, 60), (28, 50)),   # contrabass
    44: ((40, 96), (48, 88)),   # tremolo strings
    45: ((40, 96), (48, 88)),   # pizzicato strings
    46: ((24, 103), (28, 96)),  # orchestral harp
    47: ((36, 81), (40, 76)),   # timpani (tuned)
    # ── 48-55 ensemble / voice ──
    48: ((40, 96), (48, 88)),   # string ensemble 1
    49: ((40, 96), (48, 88)),   # string ensemble 2
    50: ((40, 96), (48, 88)),   # synth strings 1
    51: ((40, 96), (48, 88)),   # synth strings 2
    52: ((40, 84), (43, 79)),   # choir aahs
    53: ((40, 84), (43, 79)),   # voice oohs
    54: ((40, 84), (43, 79)),   # synth voice
    55: ((48, 88), (52, 81)),   # orchestra hit
    # ── 56-63 brass ──
    56: ((52, 84), (54, 79)),   # trumpet
    57: ((40, 72), (40, 67)),   # trombone
    58: ((28, 58), (30, 53)),   # tuba
    59: ((52, 82), (54, 77)),   # muted trumpet
    60: ((35, 77), (41, 72)),   # french horn
    61: ((48, 84), (52, 79)),   # brass section
    62: ((40, 88), (48, 81)),   # synth brass 1
    63: ((40, 88), (48, 81)),   # synth brass 2
    # ── 64-71 reed ──
    64: ((49, 81), (52, 76)),   # soprano sax
    65: ((44, 76), (49, 72)),   # alto sax
    66: ((40, 72), (44, 67)),   # tenor sax
    67: ((34, 67), (37, 62)),   # baritone sax
    68: ((58, 91), (60, 86)),   # oboe
    69: ((52, 84), (55, 79)),   # english horn
    70: ((34, 72), (36, 67)),   # bassoon
    71: ((50, 91), (52, 84)),   # clarinet
    # ── 72-79 pipe ──
    72: ((60, 96), (62, 91)),   # piccolo (sounds high)
    73: ((59, 96), (62, 91)),   # flute
    74: ((54, 91), (57, 86)),   # recorder
    75: ((48, 79), (50, 74)),   # pan flute
    76: ((48, 84), (52, 79)),   # blown bottle
    77: ((53, 84), (57, 79)),   # shakuhachi
    78: ((60, 91), (62, 86)),   # whistle
    79: ((60, 96), (64, 91)),   # ocarina
    # ── 80-87 synth lead ──
    80: ((48, 96), (52, 89)),   # lead 1 (square)
    81: ((48, 96), (52, 89)),   # lead 2 (sawtooth)
    82: ((54, 91), (57, 86)),   # lead 3 (calliope)
    83: ((54, 91), (57, 86)),   # lead 4 (chiff)
    84: ((48, 96), (52, 89)),   # lead 5 (charang)
    85: ((48, 96), (52, 89)),   # lead 6 (voice)
    86: ((48, 96), (52, 89)),   # lead 7 (fifths)
    87: ((36, 84), (40, 76)),   # lead 8 (bass+lead)
    # ── 88-95 synth pad ──
    88: ((36, 96), (43, 88)),   # pad 1 (new age)
    89: ((36, 96), (43, 88)),   # pad 2 (warm)
    90: ((36, 96), (43, 88)),   # pad 3 (polysynth)
    91: ((36, 96), (43, 88)),   # pad 4 (choir)
    92: ((36, 96), (43, 88)),   # pad 5 (bowed)
    93: ((36, 96), (43, 88)),   # pad 6 (metallic)
    94: ((36, 96), (43, 88)),   # pad 7 (halo)
    95: ((36, 96), (43, 88)),   # pad 8 (sweep)
    # ── 96-103 synth effects (wide, practical) ──
    96: ((36, 96), (43, 88)),   # FX 1 (rain)
    97: ((36, 96), (43, 88)),   # FX 2 (soundtrack)
    98: ((48, 100), (53, 93)),  # FX 3 (crystal)
    99: ((36, 96), (43, 88)),   # FX 4 (atmosphere)
    100: ((48, 100), (53, 93)), # FX 5 (brightness)
    101: ((36, 96), (43, 88)),  # FX 6 (goblins)
    102: ((36, 96), (43, 88)),  # FX 7 (echoes)
    103: ((36, 96), (43, 88)),  # FX 8 (sci-fi)
    # ── 104-111 ethnic / world (idiomatic colours for this Taiwanese folk tune) ──
    104: ((48, 84), (52, 79)),  # sitar
    105: ((48, 84), (52, 79)),  # banjo
    106: ((48, 84), (52, 79)),  # shamisen
    107: ((48, 84), (52, 79)),  # koto
    108: ((53, 89), (57, 84)),  # kalimba
    109: ((48, 84), (52, 79)),  # bagpipe
    110: ((55, 91), (55, 86)),  # fiddle (erhu-like, violin-ish range)
    111: ((58, 89), (60, 84)),  # shanai / suona (double-reed, oboe-ish)
    # ── 112-119 percussive (tuned-ish, practical bands) ──
    112: ((60, 91), (64, 86)),  # tinkle bell
    113: ((48, 84), (52, 79)),  # agogo
    114: ((48, 84), (52, 79)),  # steel drums
    115: ((48, 84), (52, 79)),  # woodblock
    116: ((40, 79), (43, 74)),  # taiko drum
    117: ((40, 79), (43, 74)),  # melodic tom
    118: ((40, 84), (43, 79)),  # synth drum
    119: ((48, 96), (52, 89)),  # reverse cymbal (FX)
    # 120-127 sound-effects programs fall to the generic range via the fallback.
}
_GENERIC_RANGE = ((24, 100), (33, 91))


def inst_range(program: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(PLAY, COMF) MIDI range for a GM program, with a family fallback."""
    if program in _INSTRUMENT_RANGES:
        return _INSTRUMENT_RANGES[program]
    # family bucket fallback (GM groups of 8)
    if 0 <= program <= 7:        # pianos
        return ((21, 108), (28, 100))
    if 24 <= program <= 31:      # guitars
        return ((40, 88), (40, 79))
    if 32 <= program <= 39:      # basses
        return ((28, 67), (28, 55))
    if 40 <= program <= 47:      # strings / orchestral
        return ((40, 96), (48, 88))
    if 56 <= program <= 63:      # brass
        return ((40, 84), (45, 79))
    if 64 <= program <= 79:      # reed + pipe (winds)
        return ((50, 91), (55, 84))
    return _GENERIC_RANGE


# ── parsed melody ────────────────────────────────────────────────────────────────────
@dataclass
class Melody:
    pitches: list[int]
    durations: list[int]          # eighth-note counts
    onsets16: list[int]           # cumulative onset of each note, in sixteenths
    durs16: list[int]             # each note's duration in sixteenths
    tempo_bpm: float
    grid_per_beat: int
    key: str
    tonic_pc: int
    phrases: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.pitches)

    @property
    def total16(self) -> int:
        return int(sum(self.durs16))


def load_melody(path: str | Path | None = None) -> Melody:
    """Load the immutable ground-truth melody from ``melody.json``."""
    p = Path(path) if path else (_HERE / "melody.json")
    if not p.is_file():
        raise ValueError(f"melody file missing: {p}")
    d = json.loads(p.read_text())
    notes = d["notes"]
    durs = [int(nt["dur_8th"]) for nt in notes]
    durs16 = [dd * 2 for dd in durs]            # 1 eighth = 2 sixteenths
    onsets16, t = [], 0
    for dd in durs16:
        onsets16.append(t)
        t += dd
    return Melody(
        pitches=[int(nt["midi"]) for nt in notes],
        durations=durs,
        onsets16=onsets16,
        durs16=durs16,
        tempo_bpm=float(d.get("tempo_bpm", 88)),
        grid_per_beat=int(d.get("grid_per_beat", 2)),
        key=str(d.get("key", "C major")),
        tonic_pc=int(d.get("tonic_pc", 0)),
        phrases=list(d.get("phrases", [])),
    )


# ── one sounded note (a chord expands to several at the same onset) ────────────────────
@dataclass(frozen=True)
class Note:
    part: int            # index into ParsedSolution.parts
    program: int         # GM program 0..127
    role: str            # "melody" | "harmony" | "bass" | "counter" | "color"
    onset16: int
    dur16: int
    pitch: int           # MIDI pitch

    @property
    def end16(self) -> int:
        return self.onset16 + self.dur16


@dataclass
class PartInfo:
    index: int
    name: str
    program: int
    role: str


@dataclass
class ParsedSolution:
    notes: list[Note]                 # PITCHED notes only, canonicalized, sorted (onset16, pitch)
    parts: list[PartInfo]
    melody: Melody
    melody_skeleton: list[int]        # the carrier's sounding pitch at each ground-truth note
    carrier_part: list[int]           # which part carries each ground-truth note
    carrier_octaves: list[int]        # per-phrase octave offset (×12) of the carrier
    tempo_bpm: float


_VALID_ROLES = ("melody", "harmony", "bass", "counter", "color", "accomp")
MIN_PITCH, MAX_PITCH = 12, 108
MIN_PARTS, MAX_PARTS = 2, 16


def _as_int(x, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or int(x) != x:
        raise ValueError(f"{what} must be an integer, got {x!r}")
    return int(x)


# ── legacy SATB ``{"voices": {...}}`` → parts conversion ───────────────────────────────
def _voices_to_parts(voices_raw: dict, melody: Melody) -> list[dict]:
    """Convert the legacy SATB chorale shape into the ``parts`` schema so a four-part
    homorhythmic (or figured) chorale is literally one instance of an arrangement. Each
    voice becomes a choir part; bare-int slots are whole-slot notes, ``[[pitch,dur16],
    ...]`` figures expand to their notes. The voice whose downbeat skeleton traces the
    melody at a constant whole octave is tagged ``role:"melody"`` (the carrier); the
    others ``role:"harmony"``."""
    if not isinstance(voices_raw, dict):
        raise ValueError("'voices' must be an object {'S':[...],'A':[...],'T':[...],'B':[...]}")
    n = melody.n
    slot_dur16 = [d * 2 for d in melody.durations]
    voice_notes: dict[str, list[tuple[int, int, int]]] = {}  # voice -> [(on16,dur16,pitch)]
    skeleton: dict[str, list[int]] = {}
    for v in ("S", "A", "T", "B"):
        seq = voices_raw.get(v)
        if not isinstance(seq, list) or len(seq) != n:
            raise ValueError(
                f"legacy voice {v!r} must be a list of exactly {n} slots (one per "
                f"melody note); got {type(seq).__name__} "
                f"len={len(seq) if isinstance(seq, list) else 'n/a'}")
        notes: list[tuple[int, int, int]] = []
        skel: list[int] = []
        for i, slot in enumerate(seq):
            on0 = melody.onsets16[i]
            if isinstance(slot, bool):
                raise ValueError(f"legacy voice {v!r} slot {i} is a bool, not a pitch")
            if isinstance(slot, (int, float)) and int(slot) == slot:
                p = int(slot)
                notes.append((on0, slot_dur16[i], p))
                skel.append(p)
                continue
            if isinstance(slot, list) and slot:
                t = on0
                total = 0
                first = None
                for pair in slot:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        raise ValueError(
                            f"legacy voice {v!r} slot {i}: each figure note must be "
                            f"[pitch,dur16]; got {pair!r}")
                    pitch, dur = pair
                    pitch = _as_int(pitch, f"legacy voice {v!r} slot {i} pitch")
                    dur = _as_int(dur, f"legacy voice {v!r} slot {i} dur16")
                    if dur <= 0:
                        raise ValueError(f"legacy voice {v!r} slot {i}: dur16 must be > 0")
                    notes.append((t, dur, pitch))
                    if first is None:
                        first = pitch
                    t += dur
                    total += dur
                if total != slot_dur16[i]:
                    raise ValueError(
                        f"legacy voice {v!r} slot {i}: figure durations sum to {total} "
                        f"sixteenths but the melody note is {slot_dur16[i]} - fill the slot")
                skel.append(first)
                continue
            raise ValueError(
                f"legacy voice {v!r} slot {i} must be a MIDI pitch or a non-empty "
                f"[[pitch,dur16],...] figure; got {slot!r}")
        voice_notes[v] = notes
        skeleton[v] = skel

    # which voice(s) carry the melody at a constant whole octave?
    carrier_voice = None
    for v in ("S", "A", "T", "B"):
        diffs = {skeleton[v][i] - melody.pitches[i] for i in range(n)}
        if len(diffs) == 1 and next(iter(diffs)) % 12 == 0:
            carrier_voice = v
            break
    # build parts (all choir aahs; a cappella chorale is a legitimate texture). If no
    # carrier voice was found we still emit the parts and let the carrier gate reject it
    # with its clear message.
    parts = []
    for v in ("S", "A", "T", "B"):
        role = "melody" if v == carrier_voice else "harmony"
        parts.append({
            "name": f"Choir {v}", "program": 52, "role": role,
            "notes": [[on, du, pi] for (on, du, pi) in voice_notes[v]],
        })
    return parts


def parse_solution(solution: dict, melody: Melody) -> ParsedSolution:
    """Validate + CANONICALIZE a submission and enforce the hard constraints.

    Accepts the native ``{"parts":[...]}`` shape OR the legacy ``{"voices":{...}}`` SATB
    chorale (auto-converted). Raises ``ValueError`` (never a silent
    zero) on a bad shape, an out-of-range value, a non-positive duration, too few/many
    parts, two notes overlapping within one part, or a missing/wrong/mid-phrase-octave-
    jumping melody carrier. Everything downstream reads this one parsed object.
    """
    if not isinstance(solution, dict):
        raise ValueError("solution must be a JSON object")

    if "parts" not in solution and "voices" in solution:
        parts_raw = _voices_to_parts(solution["voices"], melody)
    else:
        parts_raw = solution.get("parts")
    if not isinstance(parts_raw, list) or not parts_raw:
        raise ValueError(
            "solution needs a non-empty 'parts' list (or a legacy 'voices' object). Each "
            "part is {'name':str,'program':0..127,'role':'melody'|'harmony'|'bass'|"
            "'counter'|'color','notes':[[onset16,dur16,pitch],...]} where pitch is a MIDI "
            "int or a list of ints (a chord); one part must be role:'melody'.")
    if not (MIN_PARTS <= len(parts_raw) <= MAX_PARTS):
        raise ValueError(
            f"an arrangement must have {MIN_PARTS}..{MAX_PARTS} parts; got {len(parts_raw)}")

    tempo = float(solution.get("tempo_bpm", melody.tempo_bpm) or melody.tempo_bpm)

    parts: list[PartInfo] = []
    notes: list[Note] = []
    seen: set[tuple] = set()
    per_part_intervals: dict[int, list[tuple[int, int]]] = {}
    for pidx, praw in enumerate(parts_raw):
        if not isinstance(praw, dict):
            raise ValueError(f"part {pidx} must be an object, got {type(praw).__name__}")
        role = praw.get("role", "harmony")
        if role not in _VALID_ROLES:
            raise ValueError(f"part {pidx}: role {role!r} not in {_VALID_ROLES}")
        if role == "accomp":
            role = "harmony"
        program = _as_int(praw.get("program", 0), f"part {pidx} 'program'")
        if not (0 <= program <= 127):
            raise ValueError(f"part {pidx}: program {program} out of GM range 0..127")
        name = str(praw.get("name", f"part{pidx}"))[:40]
        parts.append(PartInfo(index=pidx, name=name, program=program, role=role))
        raw_notes = praw.get("notes")
        if not isinstance(raw_notes, list):
            raise ValueError(f"part {pidx} 'notes' must be a list of [onset16,dur16,pitch]")
        per_part_intervals[pidx] = []
        for k, nt in enumerate(raw_notes):
            if not isinstance(nt, (list, tuple)) or len(nt) != 3:
                raise ValueError(
                    f"part {pidx} note {k} must be [onset16, dur16, pitch]; got {nt!r}")
            on = _as_int(nt[0], f"part {pidx} note {k} onset16")
            du = _as_int(nt[1], f"part {pidx} note {k} dur16")
            if on < 0:
                raise ValueError(f"part {pidx} note {k}: onset16 {on} must be ≥ 0")
            if du < 1:
                raise ValueError(f"part {pidx} note {k}: dur16 {du} must be ≥ 1")
            if on + du > melody.total16:
                raise ValueError(
                    f"part {pidx} note {k}: onset16+dur16 = {on+du} exceeds the piece "
                    f"length {melody.total16} sixteenths")
            pitches = nt[2]
            if isinstance(pitches, (list, tuple)):
                plist = [_as_int(p, f"part {pidx} note {k} chord pitch") for p in pitches]
                if not plist:
                    raise ValueError(f"part {pidx} note {k}: empty chord pitch list")
            else:
                plist = [_as_int(pitches, f"part {pidx} note {k} pitch")]
            for p in plist:
                if not (MIN_PITCH <= p <= MAX_PITCH):
                    raise ValueError(
                        f"part {pidx} note {k}: pitch {p} out of MIDI [{MIN_PITCH},{MAX_PITCH}]")
                key = (pidx, on, du, p)
                if key in seen:
                    continue          # exact duplicate - canonicalize away
                seen.add(key)
                notes.append(Note(part=pidx, program=program, role=role,
                                   onset16=on, dur16=du, pitch=p))
            per_part_intervals[pidx].append((on, on + du))

    # ── one melodic line per part: notes within a part may not OVERLAP in time, EXCEPT a
    #    deliberate chord (several pitches sharing one [onset,dur]). Forbid PARTIAL overlaps.
    for pidx, spans in per_part_intervals.items():
        uniq = sorted(set(spans))
        for (a0, a1), (b0, b1) in zip(uniq, uniq[1:]):
            if b0 < a1 and (b0, b1) != (a0, a1):
                raise ValueError(
                    f"part {pidx}: note spans [{a0},{a1}) and [{b0},{b1}) overlap - a part "
                    f"is one line (use one part per simultaneous voice, or a chord "
                    f"[onset,dur,[p1,p2,...]] for a block).")

    notes.sort(key=lambda nt: (nt.onset16, nt.pitch))

    if len(parts) < MIN_PARTS:
        raise ValueError(
            f"need at least {MIN_PARTS} parts; got {len(parts)}")

    # ── the melody-carrier gate ─────────────────────────────────────────────────────────
    skeleton, carrier_part, carrier_octs = _check_melody_carrier(notes, melody)

    return ParsedSolution(notes=notes, parts=parts, melody=melody,
                          melody_skeleton=skeleton, carrier_part=carrier_part,
                          carrier_octaves=carrier_octs, tempo_bpm=tempo)


def _phrase_of_note(melody: Melody, note_index: int) -> int:
    """Which phrase (0..) a ground-truth melody-note index belongs to."""
    for pi, ph in enumerate(melody.phrases):
        if "notes" in ph:
            a, b = ph["notes"]
            if a <= note_index < b:
                return pi
    return 0


def _check_melody_carrier(notes: list[Note], melody: Melody):
    """Onset-anchored carrier gate. For each ground-truth melody onset, find a
    ``role:"melody"`` note that BEGINS exactly there at a whole-octave transposition of
    the ground-truth pitch; the octave offset must be CONSTANT within each phrase (it may
    change only at the four phrase boundaries). Returns (skeleton pitches, carrier part
    per note, per-phrase octave offsets). Raises ``ValueError`` on any failure."""
    mel_notes = [nt for nt in notes if nt.role == "melody"]
    if not mel_notes:
        raise ValueError(
            "no role:'melody' part carries the tune - exactly one part must trace the "
            "ground-truth melody (the human song must be the structural main melody).")
    # index melody-role note onsets → the pitches starting there
    by_onset: dict[int, list[Note]] = {}
    for nt in mel_notes:
        by_onset.setdefault(nt.onset16, []).append(nt)

    n_phrases = max(1, len(melody.phrases))
    phrase_off: dict[int, int] = {}     # phrase index → octave offset, fixed once seen
    skeleton: list[int] = []
    carrier_part: list[int] = []
    for i in range(melody.n):
        on = melody.onsets16[i]
        gt = melody.pitches[i]
        cands = by_onset.get(on, [])
        ph = _phrase_of_note(melody, i)
        # prefer a candidate consistent with the octave already fixed for this phrase
        chosen = None
        for nt in cands:
            off = nt.pitch - gt
            if off % 12 != 0 or abs(off) > 24:
                continue
            if ph in phrase_off:
                if off == phrase_off[ph]:
                    chosen = nt
                    break
            else:
                chosen = nt
                # tentatively fix the phrase octave on the first acceptable carrier note
                phrase_off[ph] = off
                break
        if chosen is None:
            # diagnose: was there a note at all? wrong octave? inconsistent within phrase?
            if not cands:
                raise ValueError(
                    f"melody note {i} (ground-truth pitch {gt} at onset {on}) has no "
                    f"role:'melody' note beginning there - the carrier must sound the "
                    f"complete tune on the fixed rhythm; ornament other parts instead.")
            offs = sorted({nt.pitch - gt for nt in cands})
            octave_offs = [d for d in offs if d % 12 == 0 and abs(d) <= 24]
            if not octave_offs:
                raise ValueError(
                    f"melody note {i} (onset {on}): carrier offsets {offs} are not a whole "
                    f"number of octaves within ±24 semitones of the ground truth - the tune "
                    f"must keep its identity (offset divisible by 12).")
            raise ValueError(
                f"melody note {i} (onset {on}): the carrier changes octave WITHIN "
                f"phrase {ph} (this phrase is fixed at offset {phrase_off[ph]}, but only "
                f"{octave_offs} begin here) - the octave may change only at a phrase "
                f"boundary, never mid-phrase.")
        skeleton.append(chosen.pitch)
        carrier_part.append(chosen.part)
    carrier_octs = [phrase_off.get(p, 0) for p in range(n_phrases)]
    return skeleton, carrier_part, carrier_octs


# ── verticals: the set of pitches sounding at each distinct onset (sustain-aware) ──────
def _onsets(notes: list[Note]) -> list[int]:
    return sorted({nt.onset16 for nt in notes})


def _sounding_at(notes: list[Note], t: int) -> list[Note]:
    return [nt for nt in notes if nt.onset16 <= t < nt.end16]


def _verticals(notes: list[Note]):
    """Yield (t, [notes sounding at t]) for each distinct onset t (a held note still
    counts) - the sustain-aware sampling the harmony, spacing, and bass checks share."""
    for t in _onsets(notes):
        yield t, _sounding_at(notes, t)


def _pitches_at(notes: list[Note], t: int, exclude_role: str | None = None) -> list[int]:
    return sorted(nt.pitch for nt in notes
                  if nt.onset16 <= t < nt.end16 and (exclude_role is None or nt.role != exclude_role))


def _accomp_pitches_at(parsed: "ParsedSolution", t: int) -> list[int]:
    """Sounding pitches of the NON-carrier parts at t - the harmonic accompaniment mass.
    Spacing / mud / octave-hole checks run on THIS (not the full vertical), because a
    melody that legitimately floats an octave above the chord must not read as 'hollow';
    the orchestration spacing rules govern the harmonic mass, while the melody's height
    is judged separately by melody-salience and its support by the harmony axis."""
    carrier = set(parsed.carrier_part)
    return sorted(nt.pitch for nt in parsed.notes
                  if nt.onset16 <= t < nt.end16 and nt.part not in carrier)


# Four registral bands (the orchestral strata): bass / tenor / alto / soprano. Used by the
# orchestration "register layering" reward and the balance gate.
def _reg_band(pitch: int) -> int:
    if pitch < 48:
        return 0          # bass (below C3)
    if pitch < 60:
        return 1          # tenor (C3..B3)
    if pitch < 72:
        return 2          # alto / mid (C4..B4)
    return 3              # soprano (C5+)


# ── extended chord recognition (pitch-class-set, C major + colour + chromatic) ─────────
@dataclass
class Vertical:
    pcs: frozenset
    label: str
    root_pc: int | None
    quality: str
    diatonic: bool
    chord_score: float
    has_third: bool


def _min_adjacent_semitone(pcs: frozenset) -> int:
    s = sorted(pcs)
    best = 12
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            d = (s[j] - s[i]) % 12
            d = min(d, 12 - d)
            best = min(best, d)
    return best


def _tertian_root_any(pcs: frozenset) -> int | None:
    for root in range(12):
        for tmpl in _TERTIAN_TEMPLATES:
            tones = {(root + t) % 12 for t in tmpl}
            if pcs.issubset(tones) and len(pcs) >= 3:
                return root
    return None


# Precompute, for every root and template, the PC-set → (label, root, quality, score,
# has_third). Built once at import (deterministic, no RNG).
def _build_chord_index():
    idx: dict[frozenset, tuple] = {}
    for root in range(12):
        for ivs, qual, sc, has_third in _CHORD_TEMPLATES:
            pcs = frozenset((root + iv) % 12 for iv in ivs)
            in_key = pcs.issubset(SCALE_SET)
            # a colour chord in-key keeps its score; out-of-key (secondary/borrowed) is
            # musical but mildly discounted.
            score = sc if in_key else max(0.62, sc - 0.30)
            roman = _ROMAN_BY_ROOT.get(root)
            if in_key and roman is not None and qual in ("maj", "min", "dim"):
                label = roman
            elif in_key and roman is not None:
                label = f"{roman}{_QUAL_SUFFIX.get(qual, '')}"
            else:
                label = f"{_PC_NAMES[root]}{_QUAL_SUFFIX.get(qual, qual)}"
            cand = (label, root, qual, score, has_third)
            # keep the highest-scoring reading for a given PC-set
            if pcs not in idx or cand[3] > idx[pcs][3]:
                idx[pcs] = cand
    return idx


_PC_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_QUAL_SUFFIX = {
    "maj": "", "min": "m", "dim": "dim", "aug": "aug", "dom7": "7", "maj7": "maj7",
    "min7": "m7", "m7b5": "m7b5", "dim7": "dim7", "maj6": "6", "min6": "m6",
    "sus2": "sus2", "sus4": "sus4", "add9": "add9", "power": "5",
}
_CHORD_INDEX = _build_chord_index()


def classify_pcs(pitches: list[int]) -> Vertical:
    """Classify a set of sounding pitches into a chord label + harmony score in [0,1].

    Pure pitch-class-set logic. A lone pitch / bare octave is harmonically empty; an
    exact recognized chord (triad, 7th, 6th, sus, add9, or their chromatic/secondary
    forms) scores high; an incomplete diatonic triad gets partial credit; a dissonant
    cluster (an adjacent minor second with no tertian reading) is punished hard."""
    pcs = frozenset(p % 12 for p in pitches)
    if len(pcs) <= 1:
        return Vertical(pcs, "unison", (next(iter(pcs)) if pcs else None), "unison",
                        True, 0.20, False)
    diatonic = pcs.issubset(SCALE_SET)
    # 1) exact recognized chord (incl. colour + chromatic forms)
    exact = _CHORD_INDEX.get(pcs)
    if exact is not None:
        label, root, qual, score, has_third = exact
        return Vertical(pcs, label, root, qual, diatonic, score, has_third)
    # 2) incomplete diatonic triad - subset of a triad and contains its root
    best_incomplete = None
    for root, (label, tones, qual) in DIATONIC_TRIADS.items():
        if pcs.issubset(set(tones)) and root in pcs and len(pcs) >= 2:
            has_third = tones[1] in pcs
            sc = 0.82 if has_third else 0.62
            cand = (label, root, qual, sc, has_third)
            if best_incomplete is None or cand[3] > best_incomplete[3]:
                best_incomplete = cand
    if best_incomplete is not None:
        label, root, qual, sc, has_third = best_incomplete
        return Vertical(pcs, label, root, "incomplete", True, sc, has_third)
    # 3) chromatic tertian (a real chord, just out of key) vs dissonant cluster
    min_adj = _min_adjacent_semitone(pcs) if len(pcs) >= 2 else 12
    tertian_root = _tertian_root_any(pcs)
    if tertian_root is not None and min_adj >= 2:
        return Vertical(pcs, "chromatic", tertian_root % 12, "chromatic", False, 0.66, True)
    if min_adj <= 1:
        return Vertical(pcs, "cluster", None, "cluster", diatonic, 0.10, False)
    return Vertical(pcs, "nontertian", None, "nontertian", diatonic, 0.42, False)


# ── the trapezoidal band primitive: 0/floor outside, 1 across the plateau ──────────────
def _band(x: float, lo0: float, lo1: float, hi1: float, hi0: float,
          floor: float = 0.0) -> float:
    """Trapezoidal sweet-spot: every aesthetic 'more is better' quantity is a band, so
    both too-little AND too-much are penalized (the anti-gaming spine)."""
    if x <= lo0:
        return floor
    if x < lo1:
        return floor + (1.0 - floor) * (x - lo0) / (lo1 - lo0)
    if x <= hi1:
        return 1.0
    if x < hi0:
        return floor + (1.0 - floor) * (hi0 - x) / (hi0 - hi1)
    return floor


def _norm_entropy(counts: dict) -> float:
    """Normalized Shannon entropy of a histogram, in [0,1]. 0 = all one value."""
    tot = sum(counts.values())
    if tot <= 0 or len(counts) <= 1:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / tot
            h -= p * math.log(p)
    return float(h / math.log(len(counts)))


# ── per-melody-note chord views (full vertical, and accompaniment-only) ────────────────
def _melody_verticals(parsed: ParsedSolution) -> list[Vertical]:
    """The full chord (all parts) sounding at each ground-truth melody onset - what the
    listener hears against the tune; the HARMONY axis runs on this."""
    out = []
    for on in parsed.melody.onsets16:
        out.append(classify_pcs(_pitches_at(parsed.notes, on)))
    return out


def _accomp_verticals(parsed: ParsedSolution) -> list[Vertical | None]:
    """The chord implied by the NON-carrier parts under each melody onset - defines the
    harmonic rhythm (a moving melody over a frozen drone can't fake motion here)."""
    carrier_parts = set(parsed.carrier_part)
    out: list[Vertical | None] = []
    for on in parsed.melody.onsets16:
        pitches = [nt.pitch for nt in parsed.notes
                   if nt.onset16 <= on < nt.end16 and nt.part not in carrier_parts]
        out.append(classify_pcs(pitches) if pitches else None)
    return out


# ── HARMONY ────────────────────────────────────────────────────────────────────────
def _is_offkey(v: "Vertical | None") -> bool:
    """True iff the vertical contains a pitch class OUTSIDE the C-major scale - a genuine
    accidental (真·离调). We test PC membership directly rather than trusting ``v.diatonic``:
    the recognizer hard-codes ``diatonic=False`` on its 'chromatic' tertian branch even for
    sets that are entirely in-scale (e.g. {E,G,A}, an incomplete vi), so the flag would
    over-count. A dissonant in-key CLUSTER is a voicing defect, not off-key, and is handled
    by the harmony chord-quality term / voicing gate - not counted here."""
    if v is None or v.quality == "cluster":
        return False
    return any(pc not in SCALE_SET for pc in v.pcs)


def _triad_basic_score(v: "Vertical | None") -> float | None:
    """How strongly an ACCOMPANIMENT vertical outlines a REAL, in-key diatonic sonority.
    Returns None when there is nothing sounding (skip - the harmonic-rhythm gate already
    handles a missing accompaniment / drone). Every complete in-key chord - plain triad OR
    colour chord (6th / 7th / add9) - is the target and scores full; genuine off-key writing
    and clusters sink; bare/ambiguous sonorities (sus, power, incomplete) score middling.
    The point is to steer toward correct, mode-grounded harmony, NOT toward the PLAINEST
    triad on every note: a diatonic 6th or add9 is a first-class in-key chord here, so an
    optimizer is free to colour the harmony instead of stripping it back to bare triads.

    NOTE this looks at the ACCOMPANIMENT mass, not the full vertical: a gong-pentatonic
    melody note sitting on a clean triad naturally adds a 2nd/4th/6th, so the FULL chord
    reads as sus2/add9 - that is correct, not a defect; basic-triad credit must therefore
    be measured on what the accompanying instruments actually spell. 'In key' is judged by
    real PC membership, so an in-scale incomplete chord the recognizer mislabels 'chromatic'
    is not wrongly penalized; only a true accidental is."""
    if v is None:
        return None
    if v.quality == "cluster":
        return 0.10
    in_key = not any(pc not in SCALE_SET for pc in v.pcs)
    if not in_key:
        return 0.25                                   # genuine off-key / chromatic / secondary
    if v.quality in ("maj", "min") and v.has_third:
        return 1.00                                   # complete basic diatonic triad
    if v.quality in ("maj6", "min6", "add9", "maj7", "min7", "dom7"):
        return 1.00                                   # in-key COLOUR chord - first-class, equal to a triad
    if v.quality == "dim" and v.has_third:
        return 0.85                                   # vii° - diatonic but less stable
    if v.has_third:
        return 0.72                                   # in-key, root+third define a quality (incl. tertian)
    if v.quality in ("sus2", "sus4", "incomplete", "power"):
        return 0.50                                   # in-key but ambiguous / no third
    if v.quality == "unison":
        return 0.45                                   # a single accompaniment PC
    return 0.45                                       # in-key non-tertian


def _score_harmony(parsed: ParsedSolution) -> tuple[float, list[Vertical], dict]:
    verts = _melody_verticals(parsed)
    chord_q = float(np.mean([v.chord_score * _FUNCTION_WEIGHT.get(v.label, 1.0)
                             for v in verts])) if verts else 0.0
    # functional root motion (collapse repeats), tonic frame
    roots = [v.root_pc for v in verts]
    motions, prev = [], None
    for r in roots:
        if r is None:
            motions.append(0.3); prev = None; continue
        if prev is not None and r != prev:
            motions.append(_ROOT_MOTION_Q.get((r - prev) % 12, 0.5))
        prev = r
    motion_q = float(np.mean(motions)) if motions else 0.3
    frame = 0.0
    if roots and roots[0] == TONIC_PC:
        frame += 0.5
    if roots and roots[-1] == TONIC_PC:
        frame += 0.5
    functional = 0.6 * motion_q + 0.4 * frame
    cad = _score_cadences(verts, parsed)
    # melody-support: the tune's pitch class should belong to (or be a tasteful tension
    # against) the chord under it - a chord that omits/clashes the melody PC is weak.
    support = 0
    for i, v in enumerate(verts):
        mpc = parsed.melody_skeleton[i] % 12
        if mpc in v.pcs:
            support += 1
    support_frac = support / len(verts) if verts else 0.0
    # BASIC-TRIAD reward + OFF-KEY penalty, both on the ACCOMPANIMENT verticals (so the
    # pentatonic-over-triad sus/add9 colouring of the FULL chord is not mistaken for either
    # a defect or for chromaticism). triad_basics → 1.0 when the accompaniment spells plain
    # diatonic triads; offkey_frac counts accompaniment verticals with any out-of-C-major PC.
    acc = _accomp_verticals(parsed)
    basics = [s for s in (_triad_basic_score(v) for v in acc) if s is not None]
    triad_basics = float(np.mean(basics)) if basics else 0.5
    offkey = sum(1 for v in acc if _is_offkey(v))
    n_acc = sum(1 for v in acc if v is not None)
    offkey_frac = offkey / n_acc if n_acc else 0.0
    H = (0.32 * chord_q + 0.20 * triad_basics + 0.18 * functional
         + 0.12 * cad + 0.18 * support_frac)
    H *= (1.0 - 0.5 * offkey_frac)                    # genuine off-key writing scales H down
    labels = [v.label for v in verts]
    detail = {
        "chord_quality": round(chord_q, 3),
        "triad_basics": round(triad_basics, 3),
        "offkey_frac": round(offkey_frac, 3),
        "functional": round(functional, 3),
        "cadence": round(cad, 3),
        "melody_support": round(support_frac, 3),
        "progression": " ".join(labels),
        "n_clusters": sum(1 for v in verts if v.quality == "cluster"),
        "n_incomplete": sum(1 for v in verts if v.quality == "incomplete"),
        "n_color": sum(1 for v in verts if v.quality in
                       ("sus2", "sus4", "add9", "maj6", "min6", "maj7", "min7", "dom7", "chromatic")),
    }
    return float(np.clip(H, 0.0, 1.0)), verts, detail


def _phrase_end_indices(parsed: ParsedSolution) -> list[int]:
    ph = parsed.melody.phrases
    if ph:
        return [int(p["notes"][1]) - 1 for p in ph if "notes" in p]
    return [parsed.melody.n - 1]


def _score_cadences(verts: list[Vertical], parsed: ParsedSolution) -> float:
    ends = _phrase_end_indices(parsed)
    if not ends:
        return 0.0
    got = 0.0
    for e in ends:
        if e < 1 or e >= len(verts):
            continue
        last, penult = verts[e], verts[e - 1]
        s = 0.0
        if penult.root_pc in (7, 11) and last.root_pc == TONIC_PC:
            s = 1.0 if penult.root_pc == 7 else 0.85            # authentic
        elif penult.root_pc == 5 and last.root_pc == TONIC_PC:
            s = 0.6                                              # plagal
        elif penult.root_pc in (2, 9) and last.root_pc in (7, 11):
            s = 0.5                                              # half cadence (→ V)
        elif last.root_pc == TONIC_PC:
            s = 0.35                                             # at least lands on I
        got += s
    return got / len(ends)


# ── ORCHESTRATION ─────────────────────────────────────────────────────────────────
MUD_FLOOR = 48          # MIDI C3 - below here close intervals turn to "mud"


def _vertical_mud(ps: list[int]) -> float:
    pen = 0.0
    for lo, hi in zip(ps, ps[1:]):
        gap = hi - lo
        if lo < MUD_FLOOR and gap < 7:                  # a 5th or less, low
            depth = (MUD_FLOOR - lo) / 12.0
            tight = (7 - gap) / 7.0
            pen += tight * (0.5 + 0.5 * min(1.0, depth))
    return pen


def _spread_shape(ps: list[int]) -> float:
    """Overtone-spacing: adjacent gaps should be non-increasing bottom→top (wide low,
    close high). Returns the fraction of correctly-ordered adjacent gaps."""
    gaps = [hi - lo for lo, hi in zip(ps, ps[1:])]
    if len(gaps) < 2:
        return 0.6                                       # too thin to assess: neutral
    inversions = sum(1 for a, b in zip(gaps, gaps[1:]) if b > a + 2)
    return 1.0 - inversions / (len(gaps) - 1)


def _doubled_pcs(ps: list[int]) -> set:
    """Pitch classes that appear in more than one octave (true doublings)."""
    by_pc: dict[int, set] = {}
    for p in ps:
        by_pc.setdefault(p % 12, set()).add(p // 12)
    return {pc for pc, octs in by_pc.items() if len(octs) > 1}


def _doubling_score(ps: list[int], v: Vertical) -> float:
    if v.root_pc is None:
        return 0.7
    doubled = _doubled_pcs(ps)
    if not doubled:
        return 1.0
    third = (v.root_pc + (4 if "maj" in v.quality or v.quality == "dom7" else 3)) % 12
    fifth = (v.root_pc + 7) % 12
    lead = 11
    s = []
    for pc in doubled:
        if pc == lead and lead not in {v.root_pc, fifth}:
            s.append(0.15)                               # doubled leading tone: poor
        elif pc == v.root_pc:
            s.append(1.0)
        elif pc == fifth:
            s.append(0.9)
        elif pc == third and ("maj" in v.quality):
            s.append(0.55)                               # doubled major 3rd: sparingly
        elif pc == third:
            s.append(0.78)
        else:
            s.append(0.6)
    return float(np.mean(s))


def _register_fit(parsed: ParsedSolution) -> tuple[float, dict]:
    """Soft comfort: each part's notes inside the instrument's COMFORTABLE band."""
    overflow = 0.0
    total = 0
    per_part = {}
    for p in parsed.parts:
        _, (lo, hi) = inst_range(p.program)
        pn = [nt.pitch for nt in parsed.notes if nt.part == p.index]
        if not pn:
            continue
        ov = sum(max(0, lo - x) + max(0, x - hi) for x in pn)
        overflow += ov
        total += len(pn)
        per_part[p.name] = round(ov / max(1, len(pn)), 2)
    # mean per-note semitone overflow → a gentle slope (≈0.012 per semitone), like the
    # SATB voice-range term (a near-miss is a gradient, not a cliff).
    fit = max(0.0, 1.0 - 0.012 * (overflow / max(1, total)))
    return float(fit), {"mean_overflow": round(overflow / max(1, total), 2)}


def _classify_texture(parsed: ParsedSolution, lo16: int, hi16: int) -> str:
    """One phrase's texture label from the non-carrier parts' rhythmic independence."""
    carrier = set(parsed.carrier_part)
    parts_onsets: list[set] = []
    for p in parsed.parts:
        if p.index in carrier:
            continue
        ons = {nt.onset16 for nt in parsed.notes
               if nt.part == p.index and lo16 <= nt.onset16 < hi16}
        if ons:
            parts_onsets.append(ons)
    n_voices = len(parts_onsets)
    if n_voices == 0:
        return "mono"
    # rhythmic independence: mean pairwise (1 - Jaccard) of onset sets
    if n_voices >= 2:
        sims = []
        for i in range(n_voices):
            for j in range(i + 1, n_voices):
                a, b = parts_onsets[i], parts_onsets[j]
                u = len(a | b) or 1
                sims.append(1.0 - len(a & b) / u)
        indep = float(np.mean(sims)) if sims else 0.0
    else:
        indep = 0.0
    mel_onsets = sum(1 for nt in parsed.notes
                     if nt.part in carrier and lo16 <= nt.onset16 < hi16)
    acc_onsets = sum(len(s) for s in parts_onsets)
    if indep > 0.45:
        return "poly"
    if acc_onsets <= 0.6 * max(1, mel_onsets):
        return "homophonic"
    return "mel+acc"


def _score_orchestration(parsed: ParsedSolution, verts: list[Vertical]) -> tuple[float, dict]:
    # (a) registral spacing + mud + doubling, over the ACCOMPANIMENT MASS (non-carrier)
    #     with ≥2 voices - the melody's height is judged by salience, not spacing.
    spreads, muds, doublings = [], [], []
    for t in _onsets(parsed.notes):
        ps = _accomp_pitches_at(parsed, t)
        if len(ps) < 2:
            continue
        spreads.append(_spread_shape(ps))
        muds.append(_vertical_mud(ps))
        i = _onset_to_melody_index(parsed, t)
        if i is not None:
            doublings.append(_doubling_score(ps, verts[i]))
    spacing = (float(np.mean(spreads)) if spreads else 0.6) - 0.15 * (float(np.mean(muds)) if muds else 0.0)
    spacing = float(np.clip(spacing, 0.0, 1.0))
    doubling = float(np.mean(doublings)) if doublings else 1.0
    # (b) register fit
    reg_fit, reg_detail = _register_fit(parsed)
    # (c) texture variety across phrases + change at boundaries
    labels = []
    for ph in parsed.melody.phrases:
        a, b = ph["notes"]
        lo = parsed.melody.onsets16[a]
        hi = parsed.melody.onsets16[b] if b < parsed.melody.n else parsed.melody.total16
        labels.append(_classify_texture(parsed, lo, hi))
    n_distinct = len(set(labels))
    variety = _band(n_distinct, 0.9, 2, 3, 4.5, floor=0.35)
    if len(labels) >= 2:
        changes = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
        boundary = 0.4 + 0.6 * (changes / (len(labels) - 1))
    else:
        boundary = 0.4
    texture = 0.6 * variety + 0.4 * boundary
    # (d) PALETTE - the richness of the orchestration, three banded ingredients (replaces
    #     the old single distinct-program term, which plateaued at 2 programs and PENALIZED
    #     a large ensemble - so the optimum was 3 instruments). Now a varied, layered,
    #     multi-family palette is actively rewarded, while still banded so dumping 16
    #     redundant parts is not (the balance gate guards the rest):
    #       * program count - widened plateau (3..14), only gently tapering past the top;
    #       * family variety - distinct GM families (winds / strings / brass / pluck /
    #         keys / voice / world), sweet spot 3..6 (one family = monochrome → low);
    #       * register layering - how many of the four registral strata (bass/tenor/alto/
    #         soprano) the WHOLE ensemble occupies, sweet spot 3..4 (rewards orchestral depth).
    n_prog = len({p.program for p in parsed.parts})
    prog_term = _band(n_prog, 0.9, 3, 14, 17, floor=0.45)
    n_family = len({_gm_family(p.program) for p in parsed.parts})
    family_term = _band(n_family, 0.9, 4, 7, 9, floor=0.40)
    bands_used = {_reg_band(nt.pitch) for nt in parsed.notes}
    layering = _band(len(bands_used), 0.9, 3, 4, 4.5, floor=0.35)
    palette = 0.34 * prog_term + 0.34 * family_term + 0.32 * layering
    # (e) melody salience: is the tune the top (or near-top) sounding voice?
    salience = _melody_salience(parsed)
    O = (0.24 * spacing + 0.15 * doubling + 0.17 * reg_fit
         + 0.15 * texture + 0.15 * palette + 0.14 * salience)
    detail = {
        "spacing": round(spacing, 3), "mud": round(float(np.mean(muds)) if muds else 0.0, 3),
        "doubling": round(doubling, 3), "register_fit": round(reg_fit, 3),
        "texture_labels": labels, "texture_variety": round(variety, 3),
        "timbral_diversity": round(palette, 3),
        "palette": round(palette, 3), "n_programs": n_prog,
        "n_families": n_family, "family_variety": round(family_term, 3),
        "register_bands": len(bands_used), "register_layering": round(layering, 3),
        "salience": round(salience, 3),
        **reg_detail,
    }
    return float(np.clip(O, 0.0, 1.0)), detail


def _onset_to_melody_index(parsed: ParsedSolution, t: int) -> int | None:
    # exact match to a ground-truth melody onset, else None
    try:
        return parsed.melody.onsets16.index(t)
    except ValueError:
        return None


def _melody_salience(parsed: ParsedSolution) -> float:
    """Soft reward that the melody is the highest-sounding voice at its onsets (the most
    audible 'main melody' placement). Buried tune still valid (hard gate already passed),
    just lower."""
    hits = 0
    for i, on in enumerate(parsed.melody.onsets16):
        mp = parsed.melody_skeleton[i]
        others = [nt.pitch for nt in parsed.notes
                  if nt.onset16 <= on < nt.end16 and nt.part not in set(parsed.carrier_part)]
        if not others or mp >= max(others):
            hits += 1
        elif mp >= np.percentile(others, 75):
            hits += 0.5
    return float(hits / parsed.melody.n) if parsed.melody.n else 0.0


# ── INTEREST ─────────────────────────────────────────────────────────────────────
def _part_skeleton(parsed: ParsedSolution, pidx: int) -> list[int]:
    """A part's lowest sounding pitch at each ground-truth melody onset (its line), or
    None where it is silent - used for activity / independence measures."""
    out = []
    for on in parsed.melody.onsets16:
        ps = [nt.pitch for nt in parsed.notes
              if nt.part == pidx and nt.onset16 <= on < nt.end16]
        out.append(min(ps) if ps else None)
    return out


def _score_interest(parsed: ParsedSolution, verts: list[Vertical]) -> tuple[float, dict]:
    carrier = set(parsed.carrier_part)
    non_carrier = [p.index for p in parsed.parts if p.index not in carrier]
    n = parsed.melody.n

    # (a) inner-voice activity + independence vs the tune
    skel_mel = parsed.melody_skeleton
    move_rates, varieties, indeps = [], [], []
    for pidx in non_carrier:
        line = _part_skeleton(parsed, pidx)
        present = [x for x in line if x is not None]
        if len(present) < 2:
            move_rates.append(0.0); varieties.append(0.0); indeps.append(0.0); continue
        moves = sum(1 for a, b in zip(line, line[1:]) if a is not None and b is not None and a != b)
        steps = sum(1 for a, b in zip(line, line[1:]) if a is not None and b is not None)
        move_rates.append(moves / max(1, steps))
        varieties.append(min(1.0, (len(set(present)) - 1) / 5.0))
        # contrary/oblique vs the melody
        co, tot = 0, 0
        for i in range(n - 1):
            a, b = line[i], line[i + 1]
            if a is None or b is None:
                continue
            mb = skel_mel[i + 1] - skel_mel[i]
            vb = b - a
            tot += 1
            if vb == 0 or mb == 0:
                co += 1
            elif (vb > 0) != (mb > 0):
                co += 1
        indeps.append(co / max(1, tot))
    # Per-part activity scores. Blend the OVERALL mean (general liveliness) with the mean of
    # the most-active HALF of the parts (the genuine moving inner lines / counter-melodies):
    # a real orchestration mixes moving lines WITH sustained pads, so a flat mean over every
    # part would punish adding a (correct) sustained colour part. The top-lines term rewards
    # having strong independent lines without penalizing the harmonic bed.
    per_part = [0.5 * m + 0.25 * v + 0.25 * ind
                for m, v, ind in zip(move_rates, varieties, indeps)]
    if per_part:
        srt = sorted(per_part, reverse=True)
        k = max(1, len(srt) // 2)
        top_lines = float(np.mean(srt[:k]))
        activity = float(0.45 * np.mean(per_part) + 0.55 * top_lines)
    else:
        activity = 0.0

    # (b) phrase-to-phrase orchestration development (through-composed: no repeats)
    contrast = _contrast_score(parsed)

    # (c) density contour across phrases
    density_level, contour = _density_contour(parsed)

    # (d) entropy band (between monotony and noise) on non-carrier pitch-classes + rhythm
    pc_hist: dict[int, int] = {}
    rhythm_hist: dict[int, int] = {}
    for nt in parsed.notes:
        if nt.part in carrier:
            continue
        pc_hist[nt.pitch % 12] = pc_hist.get(nt.pitch % 12, 0) + 1
        rhythm_hist[nt.dur16] = rhythm_hist.get(nt.dur16, 0) + 1
    H_pitch = _norm_entropy(pc_hist)
    H_rhythm = _norm_entropy(rhythm_hist)
    entropy = _band(0.5 * H_pitch + 0.5 * H_rhythm, 0.15, 0.40, 0.78, 0.97, floor=0.2)

    # (e) figuration: off-beat (between-onset) notes treated as proper NCTs, GATING the
    # raw rhythmic vitality (motion only counts when the off-beat notes are correct).
    figuration, fig_detail = _score_figuration(parsed)
    raw_vitality, vit_detail = _rhythmic_vitality(parsed)
    vitality = raw_vitality * figuration

    # (f) 节奏对齐 (COMPLEMENTARY 答话): the accompaniment answers the tune in its
    # held-note GAPS with genuine motion, rather than striking in lockstep. A moderate
    # reward for call-and-response phrasing (a block chorale / same-note hammer earns the
    # floor - this is a composition task, not a homorhythmic chorale).
    rhythm_answer, answer_detail = _rhythm_complement(parsed)

    I = (0.24 * activity + 0.20 * contrast + 0.14 * density_level
         + 0.10 * contour + 0.10 * entropy + 0.10 * vitality + 0.12 * rhythm_answer)

    detail = {
        "inner_activity": round(activity, 3), "contrast": round(contrast, 3),
        "density_level": round(density_level, 3), "contour": round(contour, 3),
        "entropy": round(entropy, 3), "H_pitch": round(H_pitch, 3),
        "H_rhythm": round(H_rhythm, 3), "figuration": round(figuration, 3),
        "raw_vitality": round(raw_vitality, 3), "vitality": round(vitality, 3),
        **fig_detail, **vit_detail, **answer_detail,
    }
    return float(np.clip(I, 0.0, 1.0)), detail


def _phrase_spans16(parsed: ParsedSolution) -> list[tuple[int, int]]:
    spans = []
    for ph in parsed.melody.phrases:
        a, b = ph["notes"]
        lo = parsed.melody.onsets16[a]
        hi = parsed.melody.onsets16[b] if b < parsed.melody.n else parsed.melody.total16
        spans.append((lo, hi))
    return spans


def _contrast_score(parsed: ParsedSolution) -> float:
    """Reward orchestration VARIETY ACROSS the (through-composed) phrases - the
    anti-monotony principle for a 4-distinct-phrase form: a good arrangement does not run
    one identical accompaniment texture under every phrase; it develops the orchestration
    phrase to phrase. We characterize each phrase by a TEXTURE FINGERPRINT that is
    independent of the (necessarily different) pitch content - namely (a) the SET of active
    non-carrier instruments, and (b) each part's RHYTHMIC pattern (the set of relative
    onset+duration cells). Phrases that share the same instruments AND the same rhythmic
    figures are monotonous (low score); phrases that change the active palette and/or the
    rhythmic figuration are developed (high score). Pitch is deliberately excluded so a
    mere chord change is NOT mistaken for orchestration development.

    (NOTE: 望春風's verse is through-composed - four distinct phrases, not a strophic
    repeat - so this measures phrase-to-phrase DEVELOPMENT, not 'vary the repeat'.)"""
    spans = _phrase_spans16(parsed)
    if len(spans) < 2:
        return 0.6
    carrier = set(parsed.carrier_part)

    def fingerprint(lo, hi):
        active = frozenset(nt.part for nt in parsed.notes
                           if nt.part not in carrier and lo <= nt.onset16 < hi)
        rhythm = frozenset((nt.part, nt.onset16 - lo, nt.dur16) for nt in parsed.notes
                           if nt.part not in carrier and lo <= nt.onset16 < hi)
        return active, rhythm

    scores = []
    for i in range(len(spans) - 1):
        a_act, a_rhy = fingerprint(*spans[i])
        b_act, b_rhy = fingerprint(*spans[i + 1])
        if not a_rhy and not b_rhy:
            scores.append(0.3); continue
        # instrumentation change: fraction of the palette that differs between phrases
        pal_union = len(a_act | b_act) or 1
        pal_change = len(a_act ^ b_act) / pal_union
        # rhythmic-figure change: 1 − Jaccard of the (part, onset, dur) cells
        rhy_union = len(a_rhy | b_rhy) or 1
        rhy_change = len(a_rhy ^ b_rhy) / rhy_union
        change = 0.4 * pal_change + 0.6 * rhy_change
        # sweet spot: SOME development (≥~0.2 change) is good; near-zero change is
        # monotonous; total change every phrase is incoherent. Band it.
        scores.append(_band(change, 0.0, 0.18, 0.75, 1.0, floor=0.25))
    return float(np.mean(scores)) if scores else 0.6


def _density_contour(parsed: ParsedSolution) -> tuple[float, float]:
    # per-onset sounding-voice count
    dens = [len(_sounding_at(parsed.notes, t)) for t in _onsets(parsed.notes)]
    mean_d = float(np.mean(dens)) if dens else 0.0
    density_level = _band(mean_d, 1.3, 2.3, 8.0, 12.0, floor=0.3)
    # per-phrase mean density → variation (shaped contour), banded on its std
    spans = _phrase_spans16(parsed)
    per_phrase = []
    for lo, hi in spans:
        ts = [t for t in _onsets(parsed.notes) if lo <= t < hi]
        if ts:
            per_phrase.append(float(np.mean([len(_sounding_at(parsed.notes, t)) for t in ts])))
    contour_var = float(np.std(per_phrase)) if len(per_phrase) >= 2 else 0.0
    contour = _band(contour_var, 0.08, 0.35, 1.6, 3.2, floor=0.3)
    return density_level, contour


# ── figuration: off-beat (non-chord) tones properly treated (passing/neighbor/susp) ────
_NCT_REWARD = {
    "chord": 1.0, "passing": 1.0, "neighbor": 1.0, "suspension": 1.0,
    "appoggiatura": 0.8, "escape": 0.7, "unprepared": 0.1,
}


def _classify_nct(prev_p, p, next_p, chord_pcs, prev_chord_pcs) -> str:
    if p % 12 in chord_pcs:
        return "chord"
    if prev_p is None or next_p is None:
        return "unprepared"
    approach = p - prev_p
    resolution = next_p - p
    step = lambda d: 1 <= abs(d) <= 2
    if step(approach) and step(resolution) and (approach > 0) == (resolution > 0):
        return "passing"
    if step(approach) and step(resolution) and next_p == prev_p:
        return "neighbor"
    if step(resolution) and resolution < 0:
        if approach == 0 or (prev_chord_pcs and prev_p % 12 in prev_chord_pcs):
            return "suspension"
        return "appoggiatura"
    if step(approach) and abs(resolution) >= 3:
        return "escape"
    return "unprepared"


def _slot_chord_pcs(parsed: ParsedSolution, i: int) -> set:
    """Chord PCs at melody-note slot i = the full vertical at that ground-truth onset."""
    return set(p % 12 for p in _pitches_at(parsed.notes, parsed.melody.onsets16[i]))


def _score_figuration(parsed: ParsedSolution) -> tuple[float, dict]:
    """For every NON-carrier note that begins BETWEEN two ground-truth melody onsets (an
    off-beat ornament), classify it as a chord tone or a specific NCT and reward correct
    treatment. A purely on-beat (homorhythmic) submission has no off-beat notes → 1.0
    (figuration is opt-in enrichment); a texture full of unprepared dissonances → low."""
    onset_set = set(parsed.melody.onsets16)
    carrier = set(parsed.carrier_part)
    # group non-carrier notes by part, sorted, to find approach/resolution neighbors
    by_part: dict[int, list[Note]] = {}
    for nt in parsed.notes:
        if nt.part in carrier:
            continue
        by_part.setdefault(nt.part, []).append(nt)
    total_off, good = 0, 0.0
    counts: dict[str, int] = {}
    for pidx, ns in by_part.items():
        ns.sort(key=lambda x: (x.onset16, x.pitch))
        for k, nt in enumerate(ns):
            if nt.onset16 in onset_set:
                continue                          # on-beat structural note
            # which melody slot does this off-beat note fall in?
            i = _slot_index_for_onset(parsed, nt.onset16)
            if i is None:
                continue
            chord_pcs = _slot_chord_pcs(parsed, i)
            prev_chord = _slot_chord_pcs(parsed, i - 1) if i > 0 else None
            prev_p = ns[k - 1].pitch if k > 0 else None
            next_p = ns[k + 1].pitch if k + 1 < len(ns) else None
            kind = _classify_nct(prev_p, nt.pitch, next_p, chord_pcs, prev_chord)
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "chord":
                continue
            total_off += 1
            good += _NCT_REWARD[kind]
    if total_off == 0:
        return 1.0, {"figure_notes": 0, "nct_types": _fmt_counts(counts)}
    return float(good / total_off), {"figure_notes": total_off, "nct_types": _fmt_counts(counts)}


def _fmt_counts(counts: dict) -> str:
    return " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def _slot_index_for_onset(parsed: ParsedSolution, t: int) -> int | None:
    """The melody-note slot index whose [onset, next_onset) contains time t."""
    ons = parsed.melody.onsets16
    for i in range(len(ons)):
        nxt = ons[i + 1] if i + 1 < len(ons) else parsed.melody.total16
        if ons[i] <= t < nxt:
            return i
    return None


def _rhythmic_vitality(parsed: ParsedSolution) -> tuple[float, dict]:
    """Reward complementary off-beat motion (answering the tune in its gaps) over a stiff
    block, banded. Gated by figuration correctness in the caller.

    Off-beat credit is given ONLY to notes that land on the EIGHTH-note grid
    (``onset16 % 2 == 0``): the tune already moves in eighths, so the natural complementary
    motion is eighth-note answers, not sixteenth-note flurries. A note on an odd sixteenth
    earns no vitality credit (and is additionally penalized by ``G_pace``) - the gradient
    points at calm, singable motion, not busier subdivision."""
    onset_set = set(parsed.melody.onsets16)
    carrier = set(parsed.carrier_part)
    accomp = [nt for nt in parsed.notes if nt.part not in carrier]
    if not accomp:
        return 0.0, {"offbeat_rate": 0.0, "complementarity": 0.0}
    # an off-beat note answers the tune only if it sits on the eighth grid AND is not itself
    # a melody onset (a between-onset eighth-note motion).
    offbeat = sum(1 for nt in accomp if nt.onset16 not in onset_set and nt.onset16 % 2 == 0)
    offbeat_rate = offbeat / max(1, len(accomp))
    # complementarity: melody slots where SOME accompaniment moves (on the eighth grid) in the gap
    n = parsed.melody.n
    gap_fills = 0
    for i in range(n):
        lo = parsed.melody.onsets16[i]
        hi = parsed.melody.onsets16[i + 1] if i + 1 < n else parsed.melody.total16
        if any(lo < nt.onset16 < hi and nt.onset16 % 2 == 0 for nt in accomp):
            gap_fills += 1
    complementarity = gap_fills / max(1, n)
    score = 0.5 * min(1.0, offbeat_rate * 2.5) + 0.5 * complementarity
    return float(min(1.0, score)), {"offbeat_rate": round(offbeat_rate, 3),
                                     "complementarity": round(complementarity, 3)}


def _rhythm_complement(parsed: ParsedSolution) -> tuple[float, dict]:
    """节奏对齐 as COMPLEMENTARY 答话 - reward the accompaniment for ANSWERING the tune in
    its held-note GAPS with genuine motion, NOT for striking in lockstep with it.

    A musical accompaniment dovetails with the melody: while the tune SUSTAINS a long note
    the accompaniment fills the space with a moving line (a passing figure, an arpeggio, an
    inner-voice answer), and where the tune is already busy it steps back. This is the
    complementary-rhythm / call-and-response principle - the opposite of a homorhythmic
    block chorale.

    For every HELD melody slot (duration ≥ a quarter, so there is room for an eighth-note
    answer), we ask: does SOME non-carrier part place a NEW onset strictly inside the gap,
    on the EIGHTH grid, that actually MOVES (its pitch differs from that part's immediately
    preceding note)? The reward is the banded fraction of held slots so answered.

    Why this is ungameable and fixes the earlier density-echo term:
      * a block-chord CHORALE that re-strikes the whole chord on every melody onset places
        NO note inside the gaps → 0 answers → floor (we no longer reward striking WITH the
        tune - a chorale is not the target of this task);
      * a 同音连敲 hammer (the same chord re-articulated every eighth under a held note) has
        onsets in the gap but NO pitch motion → not counted → floor (re-articulating one
        pitch is not idiomatic accompaniment motion);
      * a frozen DRONE has no onsets in the gaps at all → floor;
      * only a genuinely MOVING line in the tune's gaps scores - which is what winds,
        strings, plucked and keyboard accompaniments actually play.

    Banded with a sweet spot: answering SOME gaps is the goal; answering literally every
    gap is relentless (the tune should still breathe), so the top of the band tapers."""
    mel = parsed.melody
    n = mel.n
    carrier = set(parsed.carrier_part)
    # For each non-carrier part, the pitch-SET struck at each of its distinct onset times
    # (a chord is one set). "Motion" = a part's pitch-set CHANGES from its previous onset;
    # re-striking the SAME chord (a 同音/同和弦连敲 hammer) is NOT motion.
    part_onsets: dict[int, dict[int, frozenset]] = {}
    for nt in parsed.notes:
        if nt.part in carrier:
            continue
        d = part_onsets.setdefault(nt.part, {})
        d[nt.onset16] = d.get(nt.onset16, frozenset()) | {nt.pitch}
    if not part_onsets:
        return 0.0, {"rhythm_answer": 0.0, "answered_gaps": 0, "gap_slots": 0, "answer_frac": 0.0}
    # per part, its onset times in order (to find the immediately preceding struck chord)
    part_times = {p: sorted(d) for p, d in part_onsets.items()}
    GAP_MIN = 4                               # a quarter note or longer = room for an 8th answer
    gap_slots, answered = 0, 0
    for i in range(n):
        lo = mel.onsets16[i]
        hi = mel.onsets16[i + 1] if i + 1 < n else mel.total16
        if hi - lo < GAP_MIN:
            continue                          # an eighth-note slot has no room to answer
        gap_slots += 1
        found = False
        for p, times in part_times.items():
            sets = part_onsets[p]
            for j, t in enumerate(times):
                if not (lo < t < hi):
                    continue                  # onset must be STRICTLY inside the held-note gap
                if t % 2 != 0:
                    continue                  # eighth grid only (16th flurries handled by G_pace)
                prev_set = sets[times[j - 1]] if j > 0 else None
                if prev_set is None or sets[t] != prev_set:   # a real chord/pitch CHANGE, not a re-strike
                    found = True
                    break
            if found:
                break
        if found:
            answered += 1
    if gap_slots == 0:
        return 0.5, {"rhythm_answer": 0.5, "answered_gaps": 0, "gap_slots": 0, "answer_frac": 0.0}
    frac = answered / gap_slots
    # 0 answers → floor; rises to full by 0.35; full through 0.85; a gentle taper above so
    # answering EVERY gap (no space left for the tune) is a mild fault, never a degenerate.
    score = _band(frac, 0.05, 0.35, 0.85, 1.5, floor=0.2)
    return float(score), {"rhythm_answer": round(score, 3), "answered_gaps": answered,
                          "gap_slots": gap_slots, "answer_frac": round(frac, 3)}


# ── FOUNDATION (bass) ────────────────────────────────────────────────────────────
BASS_LO = 52            # ~E3: a real bass should regularly sound at/below here


def _bass_line(parsed: ParsedSolution) -> list[int]:
    """Lowest sounding pitch at each distinct onset (the bass the listener hears)."""
    out = []
    for t in _onsets(parsed.notes):
        ps = [nt.pitch for nt in _sounding_at(parsed.notes, t)]
        if ps:
            out.append(min(ps))
    return out


def _score_foundation(parsed: ParsedSolution, verts: list[Vertical]) -> tuple[float, dict]:
    # (i) bass presence in the bass register
    bass = _bass_line(parsed)
    if not bass:
        return 0.0, {"bass_presence": 0.0, "bass_chordtone": 0.0, "bass_motion": 0.0,
                     "inversion_frac": 0.0, "inversion_use": 0.0}
    presence = float(np.mean([1.0 if b <= BASS_LO else 0.0 for b in bass]))
    # (ii) chord-tone bass quality at each melody onset. Inversions are FIRST-CLASS: a 3rd
    # or 5th in the bass (I⁶, I⁶₄, V⁶ …) is proper voice-leading, scored on a par with a
    # root - so the arranger can walk the bass through inversions of ONE held chord instead
    # of re-rooting the harmony on every note just to earn "bass motion" (complaint 2).
    tone = []
    inv_root = inv_inversion = 0
    for i, on in enumerate(parsed.melody.onsets16):
        ps = _pitches_at(parsed.notes, on)
        if not ps:
            continue
        bpc = min(ps) % 12
        v = verts[i]
        if v.root_pc is None:
            tone.append(0.5); continue
        if bpc == v.root_pc:
            tone.append(1.0); inv_root += 1
        elif bpc == (v.root_pc + 7) % 12:
            tone.append(0.96); inv_inversion += 1        # 2nd inversion (5th in bass)
        elif bpc in ((v.root_pc + 3) % 12, (v.root_pc + 4) % 12):
            tone.append(0.94); inv_inversion += 1        # 1st inversion (3rd in bass)
        else:
            tone.append(0.0)
    bass_chordtone = float(np.mean(tone)) if tone else 0.5
    # (iii) bass motion: leaps at chord change, steps/holds within
    acc = _accomp_verticals(parsed)
    bass_at_onset = [min(_pitches_at(parsed.notes, on)) if _pitches_at(parsed.notes, on) else None
                     for on in parsed.melody.onsets16]
    good, tot = 0.0, 0
    for i in range(len(bass_at_onset) - 1):
        b0, b1 = bass_at_onset[i], bass_at_onset[i + 1]
        if b0 is None or b1 is None:
            continue
        tot += 1
        d = abs(b1 - b0)
        a0 = acc[i].pcs if acc[i] else frozenset()
        a1 = acc[i + 1].pcs if acc[i + 1] else frozenset()
        chord_change = not _fits_one_chord(a0 | a1) if (a0 and a1) else False
        if chord_change:
            good += 1.0 if d >= 3 else 0.5
        else:
            # Within a held chord, a step/hold is good - but so is a LEAP to another CHORD
            # TONE of the same harmony (arpeggiating the bass through an inversion, I → I⁶ →
            # I⁶₄). Only a leap to a NON-chord-tone under a static chord reads as aimless.
            v1 = acc[i + 1]
            on_chord_tone = (v1 is not None and v1.root_pc is not None and
                             (b1 % 12) in {v1.root_pc, (v1.root_pc + 3) % 12,
                                           (v1.root_pc + 4) % 12, (v1.root_pc + 7) % 12})
            good += 1.0 if (0 <= d <= 2 or on_chord_tone) else 0.55
    bass_motion = good / max(1, tot)
    # (iv) INVERSION USE: an arrangement that puts the root in the bass under EVERY chord
    # (all root-position) is stiff and, worse, drives the root-churn attractor (the only way
    # to get bass motion without inversions is to keep re-rooting). A tasteful mix of root-
    # position and inverted bass sits on the plateau; a 0%-inversion bass is mildly docked, a
    # >~55% all-inversions bass (rootless, unstable) rolls off too. This is a gentle band
    # (floor 0.85) - a nudge toward inversions, never a cliff.
    n_placed = inv_root + inv_inversion
    inversion_frac = (inv_inversion / n_placed) if n_placed else 0.0
    inversion_use = _band(inversion_frac, 0.0, 0.12, 0.55, 0.85, floor=0.85)
    F = (0.4 * bass_chordtone + 0.3 * presence + 0.3 * bass_motion) * inversion_use
    return float(np.clip(F, 0.0, 1.0)), {
        "bass_presence": round(presence, 3),
        "bass_chordtone": round(bass_chordtone, 3),
        "bass_motion": round(bass_motion, 3),
        "inversion_frac": round(inversion_frac, 3),
        "inversion_use": round(inversion_use, 3),
    }


def _fits_one_chord(pcs: frozenset) -> bool:
    """True if ``pcs`` sits inside a SINGLE in-key sonority - one chord held, not a change.
    Checked against the full diatonic palette (triads, 6ths, 7ths, add9, sus), not just plain
    triads: a SUSTAINED colour chord (a held I6 / IV add9 / V7) is ONE harmony, so it must not
    read as a chord CHANGE on every reattack - otherwise the harmonic-rhythm / voicing / bass-
    motion machinery would penalize the very colour chords the harmony axis now rewards. The
    frames are in-key only, so a genuine functional change (whose union spells >4 notes or an
    out-of-frame set) still counts as a change and the drone / off-key / noise detectors are
    unaffected."""
    if len(pcs) <= 1:
        return True
    for frame in _DIATONIC_CHORD_FRAMES:
        if pcs.issubset(frame):
            return True
    return False


# ── GATES (each →0 on its exploit, ~1 for real music) ──────────────────────────────────
def _gate_usage(parsed: ParsedSolution) -> tuple[float, dict]:
    """Every declared instrument must PARTICIPATE FULLY - a part that plays only one or two
    notes is dead weight. Two factors per part, multiplied:
      * COVERAGE - fraction of the melody's note-slots in which the part sounds, saturating
        at 30% (a part can earn full coverage credit by playing through roughly a third of
        the piece - e.g. one full phrase of a four-phrase form);
      * NOTE COUNT - a knee at three notes: a part with one or two notes (even one long
        sustain) reads as a 'ghost', so its credit is ``min(1, n_notes/3)``.
    The gate is MIN-SENSITIVE - ``0.35·mean + 0.65·min`` over the per-part credits - so a
    single under-participating instrument drags the whole gate down (you cannot dilute a
    ghost by averaging it against well-used parts). This kills both 'declare 8, use 1' and
    'add an instrument that only plays a note or two'; a real instrument that carries a
    whole phrase still passes comfortably."""
    n = parsed.melody.n
    onsets = parsed.melody.onsets16
    creds = []
    per_part = {}
    for p in parsed.parts:
        slots = sum(1 for on in onsets
                    if any(nt.part == p.index and nt.onset16 <= on < nt.end16
                           for nt in parsed.notes))
        cov = slots / max(1, n)
        n_notes = sum(1 for nt in parsed.notes if nt.part == p.index)
        cred = min(1.0, cov / 0.30) * min(1.0, n_notes / 3.0)
        creds.append(cred)
        per_part[p.name] = round(cred, 2)
    if creds:
        g = 0.35 * float(np.mean(creds)) + 0.65 * float(min(creds))
    else:
        g = 0.0
    return float(np.clip(g, 0.0, 1.0)), {"min_usage": round(min(creds) if creds else 0.0, 3),
                                         "mean_usage": round(float(np.mean(creds)) if creds else 0.0, 3)}


def _gate_content(parsed: ParsedSolution) -> tuple[float, dict]:
    """Non-carrier parts must add DISTINCT harmonic pitch classes, not merely double the
    tune in unison/octaves. At each melody onset, count non-melody PCs that differ from
    the melody PC; ≥2 → full. Systematic unison/octave doubling is penalized."""
    carrier = set(parsed.carrier_part)
    contents, parallel_oct, total = [], 0, 0
    for i, on in enumerate(parsed.melody.onsets16):
        mpc = parsed.melody_skeleton[i] % 12
        others = [nt.pitch for nt in parsed.notes
                  if nt.onset16 <= on < nt.end16 and nt.part not in carrier]
        nonmel = {p % 12 for p in others}
        content = len({pc for pc in nonmel if pc != mpc})
        contents.append(min(1.0, content / 1.5))     # ≥~2 distinct supporting PCs → full
        if others and all((p % 12) == mpc for p in others):
            parallel_oct += 1
        total += 1
    base = float(np.mean(contents)) if contents else 0.0
    if total:
        base *= (1.0 - 0.7 * (parallel_oct / total))
    return float(np.clip(base, 0.0, 1.0)), {"mean_content": round(base, 3),
                                            "parallel_unison_octave_onsets": parallel_oct}


def _gate_harmonic_rhythm(parsed: ParsedSolution) -> tuple[float, dict]:
    """Reward a BAND of genuine ACCOMPANIMENT chord changes that support the tune. A
    change is counted only when adjacent accompaniment PC-sets do NOT both fit one chord
    (so a moving melody over a frozen drone reads as zero changes → gate 0). Chaos (a
    change on nearly every note) ramps back down.

    A SECOND band guards the harmonic-rhythm's ROOT MOTION specifically: a healthy
    arrangement holds each ROOT for a couple of beats and moves the bass by INVERSION within
    it (I → I⁶ → IV), so the ROOT changes far less often than the surface chord does. An
    optimizer that re-roots the harmony on almost every melody note - the "root churns every
    note, every vertical a fresh root-position triad" attractor the user flagged - is pushed
    back down by ``root_pace``: the fraction of adjacent melody onsets whose ACCOMPANIMENT
    ROOT changes is banded to a calm ~1-chord-per-2-beats sweet spot (too static is already
    caught by the change band above; this end catches too-frantic root motion)."""
    acc = _accomp_verticals(parsed)
    acc_pcs = [(v.pcs if v is not None else frozenset()) for v in acc]
    n = len(acc_pcs)
    if n < 2:
        return 0.0, {"density": 0.0, "support": 0.0, "band": 0.0, "n_changes": 0,
                     "root_density": 0.0, "root_pace": 0.0}
    changes = 0
    for a, b in zip(acc_pcs, acc_pcs[1:]):
        if not a or not b:
            continue
        if not _fits_one_chord(a | b):
            changes += 1
    density = changes / (n - 1)
    band = _band(density, 0.06, 0.16, 0.62, 0.95, floor=0.0)
    if density > 0.95:
        band = 0.30
    support_hits = 0
    for i, v in enumerate(acc):
        if v is None:
            continue
        if parsed.melody_skeleton[i] % 12 in v.pcs:
            support_hits += 1
    support = support_hits / len(acc)
    # ROOT-PACE band: how often the accompaniment ROOT (not the surface voicing) changes
    # between adjacent melody onsets. A calm ~1-root-per-2-beats plan sits on the plateau;
    # re-rooting on nearly every note ramps down to a firm floor (a real taste cost, not a
    # zero - the base change-band + support already handle the truly static / off-tune ends).
    acc_roots = [(v.root_pc if v is not None else None) for v in acc]
    r_pairs = r_changes = 0
    for a, b in zip(acc_roots, acc_roots[1:]):
        if a is None or b is None:
            continue
        r_pairs += 1
        if a != b:
            r_changes += 1
    root_density = r_changes / r_pairs if r_pairs else 0.0
    root_pace = _band(root_density, 0.02, 0.08, 0.32, 0.60, floor=0.35)
    g = float(band * (0.4 + 0.6 * support) * root_pace)
    return g, {"density": round(density, 3), "n_changes": changes,
               "band": round(band, 3), "support": round(support, 3),
               "root_density": round(root_density, 3), "root_pace": round(root_pace, 3)}


def _gate_voicing(parsed: ParsedSolution, verts: list[Vertical]) -> tuple[float, dict]:
    """Close the hole the PC-set harmony cannot see - a chord can be in-key yet VOICED
    badly. Three deterministic defects on the ACCOMPANIMENT MASS (non-carrier parts):
      * chord-tone bass (inversions fine; a non-chord-tone bass is a real dissonance);
      * no static bass PEDAL - a bass that DOES NOT MOVE across accompaniment chord
        CHANGES is a drone (one held low note under shifting harmony). We measure the
        fraction of genuine chord changes over which the bass pitch stays put - NOT how
        often the bass repeats a pitch (a tonic-heavy pentatonic tune legitimately sits
        on the tonic bass under tonic harmony, which is not a pedal);
      * no OCTAVE HOLES - a ≥12-semitone gap between two ADJACENT voices ABOVE the bass
        leaves the middle empty (the bass-to-next gap is exempt: wide-low spacing is the
        correct overtone voicing)."""
    tone, holes = [], []
    bass_by_onset: dict[int, int] = {}
    onset_to_idx = {on: i for i, on in enumerate(parsed.melody.onsets16)}
    for t in _onsets(parsed.notes):
        ps = _accomp_pitches_at(parsed, t)       # assess the harmonic mass, not the melody
        if len(ps) < 2:
            continue
        bass_by_onset[t] = ps[0]
        upper_gaps = np.diff(ps[1:]) if len(ps) >= 3 else np.array([])
        holes.append(1.0 if upper_gaps.size and int(upper_gaps.max()) >= 12 else 0.0)
        i = onset_to_idx.get(t)
        v = verts[i] if i is not None else classify_pcs(ps)
        if v.root_pc is None:
            tone.append(0.5); continue
        bass_pc = ps[0] % 12
        chord_tones = {v.root_pc, (v.root_pc + 3) % 12, (v.root_pc + 4) % 12, (v.root_pc + 7) % 12}
        tone.append(1.0 if bass_pc in chord_tones else 0.0)
    if not tone:
        return 1.0, {"bass_chordtone": 1.0, "bass_pedal": 0.0, "octave_holes": 0.0}
    bass_chordtone = float(np.mean(tone))
    # PEDAL: over each genuine accompaniment chord change (at melody onsets), did the bass
    # move? A bass that never moves across changing harmony is a pedal.
    acc = _accomp_verticals(parsed)
    held_over_change, changes = 0, 0
    for i in range(parsed.melody.n - 1):
        a = acc[i].pcs if acc[i] else frozenset()
        b = acc[i + 1].pcs if acc[i + 1] else frozenset()
        if not a or not b or _fits_one_chord(a | b):
            continue                              # not a real chord change
        changes += 1
        b0 = bass_by_onset.get(parsed.melody.onsets16[i])
        b1 = bass_by_onset.get(parsed.melody.onsets16[i + 1])
        if b0 is not None and b1 is not None and b0 == b1:
            held_over_change += 1
    pedal_frac = held_over_change / changes if changes else 0.0
    pedal_pen = float(np.clip((pedal_frac - 0.45) / 0.45, 0.0, 1.0))
    hole_frac = float(np.mean(holes))
    g = bass_chordtone * (1.0 - 0.55 * pedal_pen) * (1.0 - 0.6 * hole_frac)
    return float(np.clip(g, 0.0, 1.0)), {
        "bass_chordtone": round(bass_chordtone, 3),
        "bass_pedal": round(pedal_pen, 3),
        "octave_holes": round(hole_frac, 3),
    }


def _gate_register(parsed: ParsedSolution) -> tuple[float, dict]:
    """Fraction of notes outside the instrument's PLAYABLE range; ~40% out → 0. A small
    tolerance keeps a single stray note a gradient, not a cliff."""
    bad, total = 0, 0
    worst = {}
    for p in parsed.parts:
        (lo, hi), _ = inst_range(p.program)
        pn = [nt.pitch for nt in parsed.notes if nt.part == p.index]
        b = sum(1 for x in pn if x < lo or x > hi)
        bad += b; total += len(pn)
        if b:
            worst[p.name] = b
    frac = bad / max(1, total)
    g = float(np.clip(1.0 - 2.5 * frac, 0.0, 1.0))
    return g, {"out_of_range_frac": round(frac, 3), "out_of_range_by_part": worst}


def _gate_contrast(parsed: ParsedSolution) -> tuple[float, dict]:
    """Soft gate from the phrase-development score: an arrangement that DEVELOPS its
    orchestration across the four phrases sits near ~1.0; one that runs the SAME
    accompaniment texture under every phrase (monotony) is multiplied down to ~0.45, so
    sameness is a real cost (not merely docked), while tasteful development is barely
    touched."""
    c = _contrast_score(parsed)
    g = float(np.clip(0.45 + 0.55 * c, 0.0, 1.0))
    return g, {"contrast": round(c, 3)}


def _gate_spread(parsed: ParsedSolution) -> tuple[float, dict]:
    """Registral spread must be SUSTAINED, not faked by one stratospheric note. Use the
    inter-quartile pitch range (75th−25th percentile of all sounding pitches), robust to
    a single outlier, banded: too narrow (everything in one octave) or absurdly wide both
    ramp down - but with a generous plateau so any real multi-register ensemble passes."""
    allp = [nt.pitch for nt in parsed.notes]
    if len(allp) < 4:
        return 1.0, {"iqr": 0.0}
    q75, q25 = np.percentile(allp, 75), np.percentile(allp, 25)
    iqr = float(q75 - q25)
    g = _band(iqr, 3, 8, 40, 56, floor=0.4)
    return float(g), {"iqr": round(iqr, 1)}


def _part_events(parsed: ParsedSolution, pidx: int) -> tuple[frozenset, frozenset]:
    """A part's played events for redundancy detection, as two sets:
      * ``exact`` - ``{(onset16, pitch)}`` (a same-line unison doubling shares these);
      * ``pc``    - ``{(onset16, pitch % 12)}`` (an OCTAVE-copied doubling shares THESE even
        though its exact pitches differ - closes the octave-copy evasion).
    Two parts that play (almost) the same line at the same times are near-duplicate doublings
    that add no independent voice, whatever their GM programs / families are."""
    exact, pc = set(), set()
    for nt in parsed.notes:
        if nt.part == pidx:
            exact.add((nt.onset16, nt.pitch))
            pc.add((nt.onset16, nt.pitch % 12))
    return frozenset(exact), frozenset(pc)


def _redundancy_clusters(parsed: ParsedSolution) -> tuple[int, int]:
    """Cluster the NON-CARRIER parts by played-line overlap (family-INDEPENDENT) and return
    ``(n_redundant, max_cluster)``. Two parts merge when their events overlap heavily - either
    at the exact pitch (Jaccard ≥ 0.60) or, to catch octave-copied walls, at the pitch class
    (Jaccard ≥ 0.85). ``n_redundant`` = Σ(clustersize − 1) over clusters of ≥2 parts (the
    number of duplicate lines); ``max_cluster`` = the size of the largest such cluster (a
    3+-part cluster is a same-line WALL - e.g. four instruments hammering one arpeggio - the
    exact defect the old family-hash redundancy could not see)."""
    carrier = set(parsed.carrier_part)
    ncs = [p.index for p in parsed.parts if p.index not in carrier]
    ev = {i: _part_events(parsed, i) for i in ncs}
    ids = [i for i in ncs if len(ev[i][0]) >= 2]   # a 1-note ghost cannot form a wall (usage guards it)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def jac(a, b):
        return len(a & b) / (len(a | b) or 1)

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            ia, ib = ids[a], ids[b]
            if jac(ev[ia][0], ev[ib][0]) >= 0.60 or jac(ev[ia][1], ev[ib][1]) >= 0.85:
                parent[find(ia)] = find(ib)
    sizes: dict[int, int] = {}
    for i in ids:
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    n_redundant = sum(s - 1 for s in sizes.values() if s >= 2)
    max_cluster = max(sizes.values()) if sizes else 0
    return n_redundant, max_cluster


def _gate_balance(parsed: ParsedSolution) -> tuple[float, dict]:
    """Anti-gaming gate for the WIDENED action space (up to 16 parts): a larger ensemble is
    only rewarded if the parts are genuinely DIFFERENTIATED and the texture is not clutter.
    It guards the two ways "more instruments" could be farmed once the palette terms reward
    size - without biting a real arrangement or the legacy 4-voice chorale:

      * REDUNDANCY - near-duplicate parts, detected by PLAYED-LINE OVERLAP and INDEPENDENT of
        GM family (``_redundancy_clusters``): parts that play the same events at the same
        times (exact-pitch OR pitch-class, to catch octave copies) cluster together. A few
        doublings are idiomatic (octave reinforcement), so a size-scaled tolerance is allowed;
        a pile of clones is padding. **A cluster of ≥3 identical lines is a same-line WALL** -
        e.g. four instruments (in four different families/timbres) all hammering one arpeggio,
        which the old family-hash redundancy could NOT see - and is a HARD fault (0.5 per
        extra clone in the biggest wall) even under the tolerance. Small ensembles (≤4 parts,
        e.g. an SATB chorale or a trio) are EXEMPT - they cannot pad.
      * CLUTTER - mean simultaneous sounding voices across the piece. A tasteful texture
        sits in a band; a wall where every one of 16 parts sounds on every onset is mud,
        ramped down past the band.
    Returns ~1.0 for good / the chorale / any real arrangement, low for a clones / clutter /
    same-line-wall degenerate."""
    pitched = list(parsed.parts)
    n = len(pitched)
    # (a) redundancy - family-independent, played-line overlap
    n_acc = sum(1 for p in pitched if p.index not in set(parsed.carrier_part))
    n_redundant, max_cluster = _redundancy_clusters(parsed)
    if n <= 4:
        redundancy_pen = 0.0
        n_redundant = 0                            # small ensembles cannot pad → exempt
    else:
        tol = max(1, n_acc // 4)                    # allow ~25% reinforcement doublings
        redundancy_pen = float(np.clip((n_redundant - tol) / max(1, n_acc - tol), 0.0, 1.0))
        # a same-line WALL (≥3 instruments playing one identical line, any timbres) is a hard
        # fault regardless of the tolerance: 3→0.5, 4→1.0. This is the 同音伴奏 exploit.
        if max_cluster >= 3:
            redundancy_pen = max(redundancy_pen, 0.5 * (max_cluster - 2))
        redundancy_pen = float(np.clip(redundancy_pen, 0.0, 1.0))
    # (b) clutter: mean simultaneous voices. Plateau is generous (a real chamber-orchestra
    # texture runs ~8-10 voices) so the WIDENED action space can breathe; only a true wall
    # (every part on every onset → 12-16 voices) ramps down. A genuine 16-part score that
    # uses rests / entrances keeps its mean in-band; an all-blast 16 does not.
    dens = [len(_sounding_at(parsed.notes, t)) for t in _onsets(parsed.notes)]
    mean_d = float(np.mean(dens)) if dens else 0.0
    clutter = _band(mean_d, 0.0, 1.3, 9.5, 15.0, floor=0.2)
    g = float(np.clip((1.0 - 0.85 * redundancy_pen) * clutter, 0.0, 1.0))
    return g, {"n_redundant": n_redundant, "max_redundant_cluster": max_cluster,
               "redundancy_pen": round(redundancy_pen, 3),
               "mean_voices": round(mean_d, 2), "clutter": round(clutter, 3)}


def _gate_pace(parsed: ParsedSolution) -> tuple[float, dict]:
    """Suppress sixteenth-note busyness - a calm, singable accompaniment over a frantic one.
    The tune of 望春風 moves in eighths and quarters; an arrangement that shatters its
    accompaniment into running sixteenths sounds restless and over-cheerful, against the
    song's character.

    A NON-CARRIER note is 'sixteenth-level' if it lands off the eighth grid
    (``onset16 % 2 == 1``) OR lasts a single sixteenth (``dur16 == 1``). ``pace_frac`` is
    the fraction of non-carrier notes that are sixteenth-level. The gate is a soft band:
    1.0 while ``pace_frac ≤ 0.15`` (a little sixteenth ornament is fine), ramping down to a
    floor of 0.55 by ``pace_frac = 0.50`` and held there - so a texture built largely on
    sixteenths is clearly penalized, but never zeroed (it is a taste cost, not a degenerate).
    The carrier itself is exempt (the tune's own rhythm is fixed by the hard gate)."""
    carrier = set(parsed.carrier_part)
    accomp = [nt for nt in parsed.notes if nt.part not in carrier]
    if not accomp:
        return 1.0, {"pace_frac": 0.0, "fast_notes": 0}
    fast = sum(1 for nt in accomp if nt.onset16 % 2 == 1 or nt.dur16 == 1)
    pace_frac = fast / len(accomp)
    if pace_frac <= 0.15:
        g = 1.0
    elif pace_frac >= 0.50:
        g = 0.55
    else:
        g = 1.0 - (1.0 - 0.55) * (pace_frac - 0.15) / (0.50 - 0.15)
    return float(np.clip(g, 0.0, 1.0)), {"pace_frac": round(pace_frac, 3), "fast_notes": fast}


def _gate_continuity(parsed: ParsedSolution) -> tuple[float, dict]:
    """An instrument must sound CONTINUOUSLY across the phrases it plays in - it may enter or
    leave only at a phrase boundary, not vanish (or appear) in the MIDDLE of a phrase. This
    guards the '突兀地停' defect: an accompaniment part that drops out mid-phrase (or a whole
    block of parts cutting out together) leaves a sudden hole in the texture - the opposite of
    the shaped, phrase-to-phrase orchestration the contrast term rewards.

    Two deterministic defects over the NON-CARRIER parts (the four phrase spans are
    0-64-128-192-256; a part may cleanly start/stop at a phrase boundary or the piece
    ends):
      * ABRUPT EDGE - a part whose FIRST onset is ≥16 slots (a full bar) after the start, or
        whose LAST note ends ≥16 slots before the end, at a time NOT within 3 slots of a
        phrase boundary. Each such ragged entrance/exit costs ``0.20``.
      * DENSITY CLIFF - a sudden drop in the number of simultaneously sounding non-carrier
        voices from one bar to the next, measured only at interior points ≥3 slots from a
        phrase boundary (so a planned boundary change is not penalized). A drop of >1.2 mean
        voices ramps the gate down (a whole block of parts stopping at once - the exact P4
        dropout in the gamed run).
    A part that plays a full phrase and rests a full phrase (a build/arc) enters and leaves at
    boundaries → no penalty; a genuine chamber arrangement sits at 1.0."""
    carrier = set(parsed.carrier_part)
    total = parsed.melody.total16
    spans = _phrase_spans16(parsed)
    bounds = set([lo for lo, _ in spans] + [hi for _, hi in spans] + [0, total])

    def near_boundary(t: int, tol: int = 3) -> bool:
        return any(abs(t - b) <= tol for b in bounds)

    # (a) abrupt entrances/exits mid-phrase
    abrupt = 0
    for p in parsed.parts:
        if p.index in carrier:
            continue
        pn = [nt for nt in parsed.notes if nt.part == p.index]
        if not pn:
            continue
        first = min(nt.onset16 for nt in pn)
        last = max(nt.end16 for nt in pn)
        if first >= 16 and not near_boundary(first):
            abrupt += 1
        if total - last >= 16 and not near_boundary(last):
            abrupt += 1

    # (b) mid-phrase density cliff on the eighth grid (non-carrier sounding-voice count)
    def voices_at(t: int) -> int:
        return len({nt.part for nt in parsed.notes
                    if nt.onset16 <= t < nt.end16 and nt.part not in carrier})

    cliff = 0.0
    for t in range(8, total - 8, 2):
        if near_boundary(t, 3):
            continue
        before = float(np.mean([voices_at(u) for u in range(t - 8, t, 2)]))
        after = float(np.mean([voices_at(u) for u in range(t, t + 8, 2)]))
        cliff = max(cliff, before - after)

    g = max(0.0, 1.0 - 0.20 * abrupt)
    if cliff > 1.2:
        g *= _band(cliff, 0.0, 1.2, 2.5, 4.0, floor=0.35)
    return float(np.clip(g, 0.0, 1.0)), {"abrupt_edges": abrupt, "density_cliff": round(cliff, 2)}


# ── the gated total ────────────────────────────────────────────────────────────────
# HARMONY is the PRIMARY axis (the user's first aim: correct harmony above all). The other
# three axes still matter, but a submission wins chiefly by harmonizing the tune well, in
# key, with basic diatonic chords - not by farming orchestral colour or busy inner lines.
WEIGHTS = {"harmony": 0.42, "orchestration": 0.20, "interest": 0.18, "foundation": 0.20}
FLOORS = {"harmony": 0.30, "orchestration": 0.20, "interest": 0.18, "foundation": 0.20}


@dataclass
class Scored:
    score: float
    axes: dict[str, float]
    gates: dict[str, float]
    metrics: dict
    verticals: list[Vertical]
    parsed: ParsedSolution
    diagnosis: str


def score_solution(solution: dict, melody: Melody | None = None) -> Scored:
    """Score one arrangement. Raises ``ValueError`` on an invalid solution.

    This is the single source of truth for the score, so any caller that reports a
    score goes through THIS and they can never disagree."""
    melody = melody or load_melody()
    parsed = parse_solution(solution, melody)          # raises ValueError if invalid

    H, verts, h_detail = _score_harmony(parsed)
    O, o_detail = _score_orchestration(parsed, verts)
    I, i_detail = _score_interest(parsed, verts)
    F, f_detail = _score_foundation(parsed, verts)

    g_usage, gu = _gate_usage(parsed)
    g_content, gc = _gate_content(parsed)
    g_hr, ghr = _gate_harmonic_rhythm(parsed)
    g_voic, gv = _gate_voicing(parsed, verts)
    g_reg, gr = _gate_register(parsed)
    g_contr, gco = _gate_contrast(parsed)
    g_spread, gs = _gate_spread(parsed)
    g_bal, gb = _gate_balance(parsed)
    g_pace, gp = _gate_pace(parsed)
    g_cont, gct = _gate_continuity(parsed)

    axes = {"harmony": H, "orchestration": O, "interest": I, "foundation": F}
    base = sum(WEIGHTS[k] * axes[k] for k in WEIGHTS)
    floor_pen = 1.0
    for k, fv in FLOORS.items():
        if axes[k] < fv:
            floor_pen *= max(0.0, axes[k] / fv)
    gate = (g_usage * g_content * g_hr * g_voic * g_reg * g_contr * g_spread * g_bal
            * g_pace * g_cont * floor_pen)
    score = 100.0 * base * gate

    gates = {"usage": g_usage, "content": g_content, "harmonic_rhythm": g_hr,
             "voicing": g_voic, "register": g_reg, "contrast": g_contr,
             "spread": g_spread, "balance": g_bal, "pace": g_pace,
             "continuity": g_cont, "floor_penalty": floor_pen}
    diagnosis = _build_diagnosis(axes, gates, h_detail, o_detail, i_detail, f_detail,
                                 gu, gc, ghr, gv, gr, gco, gs, gb, gp, gct)

    metrics = {
        "harmony": round(H, 4), "orchestration": round(O, 4),
        "interest": round(I, 4), "foundation": round(F, 4), "base": round(base, 4),
        "gate_usage": round(g_usage, 4), "gate_content": round(g_content, 4),
        "gate_harmonic_rhythm": round(g_hr, 4), "gate_voicing": round(g_voic, 4),
        "gate_register": round(g_reg, 4), "gate_contrast": round(g_contr, 4),
        "gate_spread": round(g_spread, 4), "gate_balance": round(g_bal, 4),
        "gate_pace": round(g_pace, 4), "gate_continuity": round(g_cont, 4),
        "floor_penalty": round(floor_pen, 4),
        "progression": h_detail["progression"],
        "n_clusters": float(h_detail["n_clusters"]),
        "n_color_chords": float(h_detail["n_color"]),
        "triad_basics": h_detail["triad_basics"],
        "offkey_frac": h_detail["offkey_frac"],
        "melody_support": h_detail["melody_support"],
        "cadence": h_detail["cadence"],
        "texture_labels": " ".join(o_detail["texture_labels"]),
        "timbral_diversity": o_detail["timbral_diversity"],
        "palette": o_detail.get("palette", 0.0),
        "n_families": float(o_detail.get("n_families", 0)),
        "family_variety": o_detail.get("family_variety", 0.0),
        "register_bands": float(o_detail.get("register_bands", 0)),
        "register_layering": o_detail.get("register_layering", 0.0),
        "salience": o_detail["salience"],
        "mud": o_detail["mud"],
        "inner_activity": i_detail["inner_activity"],
        "contrast": i_detail["contrast"],
        "figuration": i_detail["figuration"],
        "vitality": i_detail["vitality"],
        "rhythm_answer": i_detail.get("rhythm_answer", 0.0),
        "answered_gaps": float(i_detail.get("answered_gaps", 0)),
        "gap_slots": float(i_detail.get("gap_slots", 0)),
        "figure_notes": float(i_detail.get("figure_notes", 0)),
        "nct_types": i_detail.get("nct_types", ""),
        "bass_chordtone": f_detail["bass_chordtone"],
        "bass_presence": f_detail["bass_presence"],
        "bass_motion": f_detail["bass_motion"],
        "inversion_frac": f_detail.get("inversion_frac", 0.0),
        "harmonic_density": ghr["density"],
        "root_density": ghr.get("root_density", 0.0),
        "out_of_range_frac": gr["out_of_range_frac"],
        "pace_frac": gp["pace_frac"],
        "n_parts": float(len(parsed.parts)),
        "mean_voices": gb.get("mean_voices", 0.0),
        "n_redundant_parts": float(gb.get("n_redundant", 0)),
        "max_redundant_cluster": float(gb.get("max_redundant_cluster", 0)),
        "abrupt_edges": float(gct.get("abrupt_edges", 0)),
        "density_cliff": gct.get("density_cliff", 0.0),
        "instruments": " ".join(f"{p.name}({p.program})" for p in parsed.parts),
        "n_notes": float(len(parsed.notes)),
        "carrier_octaves": " ".join(str(x) for x in parsed.carrier_octaves),
        "diagnosis": diagnosis,
    }
    return Scored(score=float(score), axes=axes, gates=gates, metrics=metrics,
                  verticals=verts, parsed=parsed, diagnosis=diagnosis)


def _build_diagnosis(axes, gates, hd, od, idl, fd, gu, gc, ghr, gv, gr, gco, gs, gb, gp, gct) -> str:
    bits: list[str] = []
    if gates.get("pace", 1.0) < 0.95:
        bits.append(
            f"TOO BUSY: {gp['pace_frac']*100:.0f}% of the accompaniment is sixteenth-level "
            f"motion (off-beat or one-sixteenth notes) - the song wants a calm, singable "
            f"accompaniment in eighths and quarters; thin out the sixteenth runs.")
    if gates["usage"] < 0.6:
        bits.append(
            f"UNDERUSED PARTS: weakest part participation {gu['min_usage']:.2f} - a declared "
            f"instrument that plays only a note or two (or barely covers the piece) is dead "
            f"weight; have every part play a real, sustained line across at least a phrase, "
            f"or remove it.")
    if gates["content"] < 0.5:
        bits.append(
            f"NO HARMONIC CONTENT: accompaniment mostly doubles the tune (content "
            f"{gc['mean_content']:.2f}, {gc['parallel_unison_octave_onsets']} unison/octave "
            f"onsets) - add real chord tones (3rds, 6ths, a distinct bass), not the "
            f"melody's own pitch.")
    if gates["harmonic_rhythm"] < 0.5:
        if ghr.get("root_pace", 1.0) < 0.7 and ghr.get("root_density", 0.0) > 0.35:
            bits.append(
                f"ROOT CHURN: the accompaniment re-roots the harmony on {ghr['root_density']*100:.0f}% "
                f"of adjacent notes (root-pace {ghr['root_pace']:.2f}) - a fresh chord on nearly "
                f"every note is restless and over-consonant; HOLD each chord for a couple of beats "
                f"and move the bass by INVERSION within it (I → I6 → IV), changing the ROOT only a "
                f"handful of times per phrase.")
        else:
            bits.append(
                f"STATIC HARMONY: chord-change density {ghr['density']:.2f}, support "
                f"{ghr['support']:.2f} - a chord held all song (or one ignoring the tune) "
                f"reads as a drone; change chords every few notes to follow the melody.")
    if fd.get("inversion_use", 1.0) < 0.9 and fd.get("inversion_frac", 1.0) < 0.05:
        bits.append(
            f"ALL ROOT-POSITION BASS: the bass sits on the chord root under every chord "
            f"(inversion fraction {fd.get('inversion_frac', 0.0):.2f}) - use INVERSIONS "
            f"(a 3rd or 5th in the bass: I6, IV6, V6) so the bass can walk through ONE held "
            f"harmony instead of forcing a root change just to move.")
    if gates["voicing"] < 0.85:
        bits.append(
            f"BAD VOICING: bass-is-chord-tone {gv['bass_chordtone']:.2f}, bass-pedal "
            f"{gv['bass_pedal']:.2f}, octave-holes {gv['octave_holes']:.2f} - put a real "
            f"chord tone (root/3rd/5th) in the bass, change it with the harmony, and fill "
            f"the middle register so no ≥octave gap rings hollow.")
    if gates["register"] < 0.85:
        bits.append(
            f"OUT OF RANGE: {gr['out_of_range_frac']*100:.0f}% of notes lie outside their "
            f"instrument's playable range ({gr['out_of_range_by_part']}) - write each part "
            f"in its real range.")
    if gates["contrast"] < 0.6:
        bits.append(
            f"MONOTONOUS: the same accompaniment texture runs under every phrase "
            f"(development {gco['contrast']:.2f}) - the verse is through-composed (4 "
            f"distinct phrases); develop the orchestration phrase to phrase (change the "
            f"active instruments, the rhythmic figure, the density).")
    if gates["spread"] < 0.6:
        bits.append(
            f"NARROW/SKEWED REGISTER: pitch IQR {gs['iqr']:.0f} semitones - spread the "
            f"ensemble across a real range (a single very-high note doesn't count).")
    if gates.get("balance", 1.0) < 0.7:
        wall = (f"; {int(gb.get('max_redundant_cluster', 0))} instruments play the SAME line "
                f"(a same-line wall - different timbres doubling one part is not orchestration)"
                if gb.get("max_redundant_cluster", 0) >= 3 else "")
        bits.append(
            f"UNBALANCED ENSEMBLE: {gb['n_redundant']} near-duplicate part(s) "
            f"(redundancy {gb['redundancy_pen']:.2f}){wall}, mean {gb['mean_voices']:.1f} "
            f"simultaneous voices (clutter {gb['clutter']:.2f}) - a bigger ensemble only "
            f"helps if each part is a DISTINCT line; give every instrument its own material, "
            f"thin an over-dense texture, and don't pad with clones or same-line doublings.")
    if gates.get("continuity", 1.0) < 0.85:
        bits.append(
            f"ABRUPT TEXTURE: {gct['abrupt_edges']} instrument(s) enter or drop out in the "
            f"MIDDLE of a phrase and the texture suddenly thins by up to {gct['density_cliff']:.1f} "
            f"voices - an accompaniment part must sound continuously through the phrases it plays "
            f"and change only at phrase boundaries (0/64/128/192/256); don't let a block of parts "
            f"cut out mid-phrase.")
    if od.get("palette", 1.0) < 0.55:
        bits.append(
            f"PALE ORCHESTRATION: {int(od.get('n_families', 0))} instrument famil(ies), "
            f"{int(od.get('register_bands', 0))}/4 registral strata used - bring in more "
            f"colour (winds + strings + brass + plucked / world instruments) and layer the "
            f"ensemble from bass to soprano.")
    if hd["n_clusters"]:
        bits.append(f"HARMONY: {hd['n_clusters']} dissonant cluster(s) under the melody.")
    if hd.get("offkey_frac", 0.0) > 0.12:
        bits.append(
            f"OFF-KEY: {hd['offkey_frac']*100:.0f}% of the accompaniment chords use notes "
            f"outside C major - harmony is the PRIMARY aim; stay in the mode (diatonic triads "
            f"and their in-key 6th/7th/add9 colours).")
    if hd.get("triad_basics", 1.0) < 0.6:
        bits.append(
            f"WEAK CHORD FOUNDATION: chord-completeness score {hd['triad_basics']:.2f} - the "
            f"accompaniment should spell COMPLETE in-key chords (a triad's root+third+fifth, or "
            f"a full 6th/7th/add9 colour chord), not bare/ambiguous sonorities (a lone power "
            f"fifth, an incomplete triad, an off-key note).")
    if axes["harmony"] < 0.5:
        bits.append(
            f"WEAK HARMONY: chord quality {hd['chord_quality']:.2f}, functional "
            f"{hd['functional']:.2f}, cadence {hd['cadence']:.2f} - use complete in-key "
            f"chords that resolve at phrase ends.")
    if axes["orchestration"] < 0.5:
        bits.append(
            f"THIN ORCHESTRATION: spacing {od['spacing']:.2f}, register-fit "
            f"{od['register_fit']:.2f}, texture variety {od['texture_variety']:.2f} - space "
            f"chords wide-low/close-high, write in idiomatic ranges, and vary the texture "
            f"between phrases.")
    if axes["interest"] < 0.5:
        bits.append(
            f"DULL: inner-voice activity {idl['inner_activity']:.2f}, contrast "
            f"{idl['contrast']:.2f}, contour {idl['contour']:.2f} - give the inner parts "
            f"independent moving lines and shape the density across the form.")
    if idl.get("rhythm_answer", 1.0) < 0.5:
        bits.append(
            f"NO ANSWER TO THE TUNE: only {idl.get('answered_gaps', 0)}/"
            f"{idl.get('gap_slots', 0)} of the melody's held-note gaps get a moving "
            f"accompaniment answer (rhythm-answer {idl.get('rhythm_answer', 0):.2f}) - while "
            f"the tune sustains a long note, fill the space with a MOVING inner line or "
            f"arpeggio (call-and-response), don't just re-strike the same chord in lockstep "
            f"or hammer one pitch.")
    if axes["foundation"] < 0.5:
        bits.append(
            f"WEAK BASS: presence {fd['bass_presence']:.2f}, chord-tone {fd['bass_chordtone']:.2f}, "
            f"motion {fd['bass_motion']:.2f} - anchor the harmony with a real bass that leaps "
            f"at chord changes and steps or walks through inversions within them.")
    if not bits:
        bits.append(
            "BALANCED: in-key, colourful chords that support and follow the tune, an "
            "idiomatically orchestrated and well-spaced texture, independent moving inner "
            "voices, well-developed orchestration across the phrases, and a firm bass. Refine "
            "voicings and cadences for the last points.")
    return " ".join(bits)


# ── rendering: arrangement → listenable audio (mp3) + MIDI + a multi-part piano-roll ──
# STRICTLY fail-soft: a missing audio lib drops that artifact, never the score (scoring
# imports ONLY numpy + stdlib). The audio path CASCADES for the best sound available:
#   1) FluidSynth + a real General-MIDI soundfont (sampled instruments - the best),
#   2) the TimGM6mb soundfont bundled inside pretty_midi (no extra download),
#   3) a polished pure-numpy synth (per-family timbre + ADSR + reverb + stereo + limiter),
#   4) nothing (drop the mp3).
import os

# GM family → additive-synth recipe for the numpy fallback (harmonic amplitudes, an ADSR
# in seconds, and a vibrato depth). Picked to read as the right instrument family.
_SR = 44100


def _gm_family(program: int) -> str:
    """Map a GM program to a TIMBRAL family - used both for the orchestration palette
    (family-variety reward) and the numpy fallback synth recipe. Covers the full GM map so
    distinct timbres (incl. the ethnic/world block) read as distinct families, not all
    'strings'."""
    if 0 <= program <= 7:
        return "keys"            # pianos
    if 8 <= program <= 15:
        return "mallet"          # celesta, glock, vibes, marimba, bells, dulcimer
    if 16 <= program <= 23:
        return "organ"           # organs / accordion / harmonica
    if 24 <= program <= 31:
        return "guitar"          # guitars
    if 32 <= program <= 39:
        return "bass"
    if 40 <= program <= 51:
        return "strings"         # solo + ensemble strings
    if 52 <= program <= 54:
        return "voice"
    if program == 55:
        return "strings"         # orchestra hit
    if 56 <= program <= 63:
        return "brass"
    if 64 <= program <= 71:
        return "reed"
    if 72 <= program <= 79:
        return "flute"           # pipes / flutes
    if 80 <= program <= 87:
        return "lead"            # synth leads
    if 88 <= program <= 95:
        return "pad"             # synth pads
    if 96 <= program <= 103:
        return "pad"             # synth FX (pad-like)
    if 104 <= program <= 111:
        return "pluck"           # ethnic / world (sitar, koto, pipa, erhu, sheng, ...)
    if 112 <= program <= 119:
        return "mallet"          # tuned percussive (steel drums, woodblock, agogo, ...)
    return "strings"


_FAMILY_SPEC = {
    #            harmonic amplitudes (a1..),                       A     D     S     R    vib
    "flute":   ([1.0, 0.18, 0.08, 0.03, 0.012],                    0.04, 0.05, 0.85, 0.10, 0.004),
    "reed":    ([1.0, 0.5, 0.32, 0.18, 0.10, 0.05],                0.05, 0.06, 0.82, 0.10, 0.004),
    "strings": ([1.0, 0.5, 0.33, 0.25, 0.18, 0.13, 0.09, 0.06],    0.10, 0.10, 0.82, 0.22, 0.006),
    "brass":   ([0.6, 1.0, 0.9, 0.7, 0.55, 0.4, 0.28, 0.18],       0.06, 0.08, 0.78, 0.12, 0.004),
    "guitar":  ([1.0, 0.6, 0.4, 0.28, 0.18, 0.1],                  0.004, 0.4, 0.0, 0.3, 0.0),
    "keys":    ([1.0, 0.55, 0.32, 0.2, 0.12, 0.07],                0.005, 0.5, 0.0, 0.35, 0.0),
    "bass":    ([1.0, 0.45, 0.22, 0.12, 0.06],                     0.02, 0.18, 0.7, 0.18, 0.0),
    "voice":   ([1.0, 0.5, 0.35, 0.2, 0.12, 0.07],                 0.10, 0.08, 0.85, 0.28, 0.006),
    "organ":   ([1.0, 0.7, 0.5, 0.7, 0.3, 0.5, 0.2],               0.02, 0.02, 0.95, 0.06, 0.0),
    "mallet":  ([1.0, 0.4, 0.7, 0.2, 0.25, 0.1],                   0.002, 0.35, 0.0, 0.3, 0.0),
    "pluck":   ([1.0, 0.65, 0.45, 0.32, 0.22, 0.14, 0.08],         0.003, 0.45, 0.0, 0.35, 0.003),
    "lead":    ([1.0, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12],          0.02, 0.06, 0.8, 0.12, 0.005),
    "pad":     ([1.0, 0.55, 0.4, 0.3, 0.22, 0.16, 0.11, 0.08],     0.18, 0.15, 0.85, 0.4, 0.004),
}


def voice_events(parsed: ParsedSolution):
    """Yield (program, role, midi, t_start_sec, dur_sec, is_carrier) for every note."""
    spb = 60.0 / parsed.tempo_bpm / parsed.melody.grid_per_beat / 2.0   # sec per 16th
    carrier = set(parsed.carrier_part)
    for nt in parsed.notes:
        yield (nt.program, nt.role, nt.pitch, nt.onset16 * spb, nt.dur16 * spb,
               nt.part in carrier)


def _build_pretty_midi(parsed: ParsedSolution):
    """Build a pretty_midi.PrettyMIDI with one instrument per part (its GM program), the
    carrier mixed a little louder. Raises on import failure (caller is fail-soft)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(parsed.tempo_bpm))
    spb = 60.0 / parsed.tempo_bpm / parsed.melody.grid_per_beat / 2.0
    carrier = set(parsed.carrier_part)
    by_part: dict[int, list] = {}
    for nt in parsed.notes:
        by_part.setdefault(nt.part, []).append(nt)
    for p in parsed.parts:
        inst = pretty_midi.Instrument(program=int(p.program), name=p.name[:30])
        is_car = p.index in carrier
        for nt in by_part.get(p.index, []):
            vel = 104 if is_car else 76
            inst.notes.append(pretty_midi.Note(
                velocity=vel, pitch=int(nt.pitch),
                start=nt.onset16 * spb, end=(nt.onset16 + nt.dur16) * spb))
        pm.instruments.append(inst)
    return pm


def _fluidsynth_wav(parsed: ParsedSolution, sr: int = _SR):
    """Render the arrangement with FluidSynth + a GM soundfont (sampled instruments).
    Returns float32 mono in [-1,1], or None if fluidsynth / a soundfont is unavailable."""
    try:
        import fluidsynth  # noqa: F401  (pyfluidsynth → libfluidsynth)
    except Exception:
        return None
    try:
        pm = _build_pretty_midi(parsed)
    except Exception:
        return None
    sf2 = os.environ.get("SATB_SF2")
    try:
        if sf2 and os.path.exists(sf2):
            import fluidsynth
            fl = fluidsynth.Synth(samplerate=float(sr))
            try:
                fl.setting("synth.reverb.active", 1)   # space/depth - the big niceness win
                fl.setting("synth.chorus.active", 0)   # chorus muddies a dense texture
                fl.setting("synth.gain", 0.6)
            except Exception:
                pass
            sfid = fl.sfload(sf2)
            if sfid < 0:
                fl.delete(); return None
            audio = pm.fluidsynth(fs=sr, synthesizer=fl, sfid=sfid, normalize=False)
            fl.delete()
        else:
            audio = pm.fluidsynth(fs=sr, normalize=True)   # bundled TimGM6mb.sf2
    except Exception:
        return None
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return None
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.89).astype(np.float32)


def _adsr(n: int, sr: int, a: float, d: float, s: float, r: float) -> "np.ndarray":
    env = np.ones(n, dtype=np.float64)
    na, nd, nr = int(a * sr), int(d * sr), int(r * sr)
    na = min(na, n); nd = min(nd, max(0, n - na))
    nr = min(nr, max(0, n - na - nd))
    i = 0
    if na > 0:
        env[:na] = np.linspace(0.0, 1.0, na); i = na
    if nd > 0:
        env[i:i + nd] = np.linspace(1.0, s, nd); i += nd
    sustain_end = n - nr
    if sustain_end > i:
        env[i:sustain_end] = s if (d > 0 or a > 0) else 1.0
    if nr > 0:
        start = env[sustain_end - 1] if sustain_end > 0 else s
        env[sustain_end:] = np.linspace(start, 0.0, n - sustain_end)
    return env


def _synth_note(midi: int, dur_sec: float, sr: int, family: str, seed: int) -> "np.ndarray":
    """One note as anti-aliased additive synthesis with a per-family ADSR + vibrato."""
    amps, A, D, S, R, vib = _FAMILY_SPEC.get(family, _FAMILY_SPEC["strings"])
    ring = dur_sec + (R if S > 0 else 0.0)
    n = max(1, int(ring * sr))
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    t = np.arange(n) / sr
    # deterministic per-note phase jitter (no RNG): hash the seed into [0,2π)
    ph0 = ((seed * 2654435761) & 0xffff) / 0xffff * 2 * math.pi
    if vib > 0:
        ramp = np.clip(t / 0.25, 0, 1)
        inst = 2 * np.pi * f * t + vib * ramp * np.sin(2 * np.pi * 5.5 * t)
    else:
        inst = 2 * np.pi * f * t
    wave = np.zeros(n)
    for k, amp in enumerate(amps, start=1):
        if k * f >= sr / 2:                      # anti-alias: skip partials over Nyquist
            break
        wave += amp * np.sin(k * inst + ph0 * (k % 3))
    # plucked/keys families decay exponentially regardless of S
    if S <= 0.0:
        wave *= np.exp(-t / max(0.12, D))
    env = _adsr(n, sr, A, D, S, R)
    return (wave * env).astype(np.float64)


def _synth_reverb(dry: "np.ndarray", sr: int) -> "np.ndarray":
    """A cheap, deterministic convolution reverb (fixed-seed exponential-decay noise IR).
    Reverb is the single biggest 'niceness' win over a dry additive synth."""
    try:
        import scipy.signal as sig
    except Exception:
        return dry
    rng = np.random.RandomState(1234)            # fixed seed → deterministic
    L = int(1.2 * sr)
    tt = np.arange(L) / sr
    ir = rng.randn(L) * np.exp(-tt / 0.35)
    wet = sig.fftconvolve(dry, ir)[:dry.size]
    return 0.78 * dry + 0.22 * (wet / (np.max(np.abs(wet)) or 1.0)) * (np.max(np.abs(dry)) or 1.0)


def _synth_wav_numpy(parsed: ParsedSolution, sr: int = _SR) -> "np.ndarray":
    """Polished pure-numpy fallback: per-family additive timbres + ADSR + vibrato, a
    convolution reverb, and a soft limiter. Deterministic. Returns float32 mono."""
    ev = list(voice_events(parsed))
    if not ev:
        return np.zeros(sr // 2, dtype=np.float32)
    pitched_end = max((t0 + d for _, _, _, t0, d, _ in ev), default=0.0)
    total = pitched_end + 1.0
    buf = np.zeros(int(total * sr) + 8, dtype=np.float64)
    fam_gain = {"flute": 0.9, "reed": 0.8, "strings": 0.7, "brass": 0.8,
                "guitar": 0.85, "keys": 0.8, "bass": 0.95, "voice": 0.8,
                "organ": 0.7, "mallet": 0.85, "pluck": 0.85, "lead": 0.75, "pad": 0.6}
    for idx, (program, role, midi, t0, d, is_car) in enumerate(ev):
        fam = _gm_family(program)
        wave = _synth_note(midi, d, sr, fam, seed=idx + midi)
        g = fam_gain.get(fam, 0.7) * (1.4 if is_car else 0.85)
        s0 = int(t0 * sr)
        end = min(len(buf), s0 + wave.size)
        buf[s0:end] += g * wave[:end - s0]
    buf = _synth_reverb(buf, sr)
    # soft limiter (smooth saturation) then normalize for headroom
    peak = float(np.max(np.abs(buf))) or 1.0
    buf = np.tanh(1.1 * buf / peak) / math.tanh(1.1)
    peak = float(np.max(np.abs(buf))) or 1.0
    return (buf / peak * 0.89).astype(np.float32)


def render_audio_mp3(parsed: ParsedSolution, melody: Melody | None = None) -> bytes | None:
    """Render the arrangement to mp3 bytes (None on any failure - fail-soft). Cascades:
    FluidSynth+soundfont → bundled soundfont → numpy synth → None."""
    try:
        import subprocess
        import tempfile
        import soundfile as sf
        import imageio_ffmpeg
    except Exception:
        return None
    wav = _fluidsynth_wav(parsed)
    if wav is None:
        try:
            wav = _synth_wav_numpy(parsed)
        except Exception:
            return None
    try:
        with tempfile.TemporaryDirectory(prefix="orch_aud_") as td:
            wp = Path(td) / "o.wav"
            mp = Path(td) / "o.mp3"
            sf.write(str(wp), wav, _SR)
            ff = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([ff, "-y", "-i", str(wp), "-b:a", "192k", str(mp)],
                           capture_output=True, timeout=120)
            if mp.is_file() and mp.stat().st_size > 0:
                return mp.read_bytes()
    except Exception:
        return None
    return None


def render_midi(parsed: ParsedSolution, melody: Melody | None = None) -> bytes | None:
    """Render the arrangement to a Standard MIDI File (bytes). None on failure."""
    try:
        import tempfile
        pm = _build_pretty_midi(parsed)
        with tempfile.TemporaryDirectory(prefix="orch_mid_") as td:
            mp = Path(td) / "o.mid"
            pm.write(str(mp))
            return mp.read_bytes()
    except Exception:
        return None


# distinct colours per part (cycled) for the piano-roll
_PART_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                "#17becf", "#8c564b", "#bcbd22", "#e377c2", "#7f7f7f",
                "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
                "#3182bd"]   # ≥16 distinct so a full 16-part ensemble legend is unambiguous


def render_score_png(scored: Scored, melody: Melody | None = None) -> bytes | None:
    """A multi-part piano-roll PNG: each part a colour, the melody carrier bold, phrase
    boundaries dashed, chord labels along the top, gate values in the title. The visual
    'why' behind the score, for quick inspection."""
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    try:
        parsed = scored.parsed
        mel = parsed.melody
        spb = 60.0 / parsed.tempo_bpm / mel.grid_per_beat / 2.0
        carrier = set(parsed.carrier_part)
        fig, ax = plt.subplots(figsize=(15, 6), dpi=110)
        # assign a colour to EVERY part up front (the legend mirrors parsed.parts order).
        prog_color = {p.index: _PART_COLORS[i % len(_PART_COLORS)]
                      for i, p in enumerate(parsed.parts)}
        allp = [nt.pitch for nt in parsed.notes] or [60]
        for nt in parsed.notes:
            is_car = nt.part in carrier
            ax.add_patch(Rectangle(
                (nt.onset16 * spb, nt.pitch - 0.45), max(nt.dur16 * spb * 0.96, 0.03), 0.9,
                facecolor=prog_color[nt.part], alpha=0.95 if is_car else 0.55,
                edgecolor="black", linewidth=1.3 if is_car else 0.3))
        # phrase boundaries
        for ph in mel.phrases:
            a, _b = ph["notes"]
            x = mel.onsets16[a] * spb
            ax.axvline(x, color="#888", ls="--", lw=0.8, alpha=0.6)
        # chord labels along the top
        labels = scored.metrics["progression"].split()
        ymax = max(max(allp), 72) + 4
        ymin = min(min(allp), 40) - 3
        for i, lab in enumerate(labels):
            if i < mel.n:
                ax.text(mel.onsets16[i] * spb + 0.02, ymax - 1.5, lab, fontsize=6,
                        color="#333", ha="left", va="top", rotation=0)
        ax.set_xlim(0, mel.total16 * spb)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("MIDI pitch")
        car_names = ",".join(parsed.parts[p].name for p in sorted(carrier))
        m = scored.metrics
        ax.set_title(
            f"Bang-chhun-hong arrangement - score {scored.score:.1f}/100 | carrier: {car_names} (bold) | "
            f"{int(m['n_parts'])} parts | H={scored.axes['harmony']:.2f} O={scored.axes['orchestration']:.2f} "
            f"I={scored.axes['interest']:.2f} F={scored.axes['foundation']:.2f} | "
            f"gates use={scored.gates['usage']:.2f} hr={scored.gates['harmonic_rhythm']:.2f} "
            f"voic={scored.gates['voicing']:.2f} contr={scored.gates['contrast']:.2f}",
            fontsize=8)
        handles = [Rectangle((0, 0), 1, 1, facecolor=prog_color[p.index],
                             alpha=0.9 if p.index in carrier else 0.55)
                   for p in parsed.parts]
        ax.legend(handles, [p.name for p in parsed.parts],
                  loc="lower right", ncol=min(4, len(parsed.parts)), fontsize=7)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


# ── engraved score (proper staff notation / 总谱) via LilyPond ─────────────────────────
# Hand-generate a LilyPond .ly (the deterministic, music21-backend-free path the research
# verified byte-identical) and call `lilypond --png`. Pitched parts become staves with a
# clef chosen by register + an instrument name. Non-dyadic durations and notes crossing a
# 4/4 barline are split and TIED. Fail-soft: a
# verovio→rsvg fallback, then None (scoring never depends on engraving).
_LY_STEP = {0: "c", 1: "cis", 2: "d", 3: "dis", 4: "e", 5: "f",
            6: "fis", 7: "g", 8: "gis", 9: "a", 10: "ais", 11: "b"}
# dur16 (sixteenths) → LilyPond duration glyph, for the dyadic + dotted values.
_LY_DUR = {16: "1", 8: "2", 4: "4", 2: "8", 1: "16", 12: "2.", 6: "4.", 3: "8.", 24: "1.", 32: "1*2"}
_BAR16 = 16          # 4/4, sixteenth grid → 16 sixteenths per bar


def _ly_pitch(midi: int) -> str:
    pc, octv = midi % 12, midi // 12
    ticks = octv - 4                                  # MIDI octave 5 == C4 → one "'"
    mark = "'" * ticks if ticks >= 0 else "," * (-ticks)
    return _LY_STEP[pc] + mark


def _split_durations(dur16: int) -> list[int]:
    """Greedy largest-first decomposition of a duration into renderable dur16 chunks.
    Callers split at beat/bar boundaries FIRST (``_split_at_beats``), so by the time a
    chunk reaches here it is ≤ one beat (4 sixteenths) and decomposes into at most a
    dotted value + a tie - no pathological tie-walls."""
    out, rem = [], dur16
    for d in (32, 24, 16, 12, 8, 6, 4, 3, 2, 1):
        while rem >= d:
            out.append(d); rem -= d
    return out or [1]


_BEAT16 = 4              # 4/4, sixteenth grid → 4 sixteenths per quarter-note beat


def _split_at_beats(on: int, dur: int) -> list[tuple[int, int]]:
    """Split [on, on+dur) so no segment crosses a 4/4 BEAT boundary UNLESS the segment is a
    clean dyadic note that starts on a beat (a half/whole note may span beats). This keeps
    syncopated agent rhythms readable: an off-beat 5/16ths reads as 'note to the next beat,
    tied to the remainder' instead of greedy 'quarter + sixteenth' that hides the beat.
    A note that both starts on a beat AND has a dyadic length (2,4,8,16 sixteenths, or the
    dotted 6,12) is left whole; everything else is cut at the next beat line."""
    segs = []
    t, end = on, on + dur
    while t < end:
        length = end - t
        on_beat = (t % _BEAT16 == 0)
        # a clean note that begins on a beat and is a renderable whole/dotted value may span
        # several beats (don't shatter a half/whole note into tied quarters).
        if on_beat and length in (2, 4, 6, 8, 12, 16, 24, 32):
            segs.append((t, length)); t = end; break
        nxt_beat = (t // _BEAT16 + 1) * _BEAT16
        seg_end = min(end, nxt_beat)
        segs.append((t, seg_end - t))
        t = seg_end
    return segs


def _split_at_barlines(on: int, dur: int) -> list[tuple[int, int]]:
    """Split [on, on+dur) into segments that do not cross a 4/4 barline."""
    segs = []
    t, end = on, on + dur
    while t < end:
        nxt_bar = (t // _BAR16 + 1) * _BAR16
        seg_end = min(end, nxt_bar)
        segs.append((t, seg_end - t))
        t = seg_end
    return segs


def _split_metric(on: int, dur: int) -> list[tuple[int, int]]:
    """Split [on, on+dur) at barlines first (mandatory in 4/4 notation), then at beat
    boundaries within each bar segment (readability). The union the engraver actually uses
    for notes and rests."""
    out = []
    for (bo, bd) in _split_at_barlines(on, dur):
        out.extend(_split_at_beats(bo, bd))
    return out


def _ly_event(pitch_tokens: str, dur16: int) -> str:
    """One event (note/chord ``pitch_tokens`` already formatted) of duration dur16, tying
    across the internal non-dyadic split. ``pitch_tokens`` is e.g. ``c'`` or ``<c' e' g'>``."""
    durs = _split_durations(dur16)
    return "~ ".join(f"{pitch_tokens}{_LY_DUR[d]}" for d in durs)


def _ly_rest(dur16: int) -> str:
    return " ".join(f"r{_LY_DUR[d]}" for d in _split_durations(dur16))


def _ly_rest_metric(on: int, dur16: int) -> str:
    """Rests for a gap [on, on+dur16), split at bar+beat boundaries so a multi-beat rest
    reads as one rest per beat region rather than a greedy glyph that hides the meter."""
    return " ".join(_ly_rest(sd) for (_so, sd) in _split_metric(on, dur16))


def _ly_part_music(notes: list[Note], total16: int) -> str:
    """Render one pitched part's note list (sorted, non-overlapping single line OR chords)
    into a LilyPond music string: rests for gaps, ties across bar+beat boundaries + non-
    dyadic durs. Chords (several notes sharing one onset/dur) become a ``<...>`` chord
    event. Splitting at beat boundaries (not just barlines) keeps syncopated rhythms
    readable instead of greedy tie-walls."""
    # group by (onset, dur) → the pitches sounding as one event
    events: dict[tuple[int, int], list[int]] = {}
    for nt in notes:
        events.setdefault((nt.onset16, nt.dur16), []).append(nt.pitch)
    toks: list[str] = []
    t = 0
    for (on, dur) in sorted(events):
        if on < t:
            continue                                  # overlap guard (shouldn't happen)
        if on > t:
            toks.append(_ly_rest_metric(t, on - t))
        pitches = sorted(events[(on, dur)])
        body = (_ly_pitch(pitches[0]) if len(pitches) == 1
                else "<" + " ".join(_ly_pitch(p) for p in pitches) + ">")
        segs = _split_metric(on, dur)
        seg_toks = [_ly_event(body, sd) for (_so, sd) in segs]
        toks.append("~ ".join(seg_toks))              # tie across the metric splits
        t = on + dur
    if t < total16:
        toks.append(_ly_rest_metric(t, total16 - t))
    return " ".join(toks)


# GM programs that conventionally read in BASS clef (low instruments). Everything else
# defaults to TREBLE; a per-part register override flips the clef only when the part sits
# more than ~a 5th away from the default's comfortable side (so a flute written one octave
# low still reads treble, but a cello playing high gets a treble passage).
_BASS_CLEF_PROGRAMS = frozenset({
    32, 33, 34, 35, 36, 37, 38, 39,    # basses
    42, 43,                            # cello, contrabass
    58,                                # tuba
    70,                                # bassoon
    87,                                # synth bass+lead
})
# Wide-range instruments (keyboards, harp, organ, guitar, ensemble strings, pads) may
# legitimately be written down into bass-clef register, so a low part flips to bass. A
# NARROW treble instrument (flute, violin, oboe, …) is ALWAYS treble - written low it just
# gets ledger lines (which correctly signals an out-of-range error the register gate bills).
_WIDE_RANGE_PROGRAMS = frozenset(
    set(range(0, 8))          # pianos
    | set(range(16, 24))      # organs / accordion
    | set(range(24, 32))      # guitars
    | {46}                    # harp
    | set(range(48, 52))      # ensemble / synth strings
    | set(range(88, 104))     # pads / FX
)


def _clef_for(notes: list[Note], program: int = -1) -> str:
    """Pick a clef from the INSTRUMENT (its GM family) first, then a register override.

    The previous version chose purely by mean pitch (<56 → bass), which wrongly engraved a
    flute / violin / oboe carrier in bass clef whenever it sat in a low octave. Now a narrow
    treble instrument ALWAYS reads treble; only a wide-range instrument (piano/harp/organ/
    guitar/ensemble) written low flips to bass, and a bass instrument written high flips to
    treble."""
    default = "bass" if program in _BASS_CLEF_PROGRAMS else "treble"
    if not notes:
        return default
    mean = float(np.mean([nt.pitch for nt in notes]))
    if default == "treble" and program in _WIDE_RANGE_PROGRAMS and mean < 52:
        return "bass"                            # a low piano/harp/guitar passage
    if default == "bass" and mean >= 60:         # a high cello/bassoon passage → treble
        return "treble"
    return default


_LY_TEMPLATE = r'''\version "2.24.0"
\paper {
  indent = 12\mm
  line-width = 190\mm
  top-margin = 6\mm
  oddHeaderMarkup = ##f  evenHeaderMarkup = ##f
  oddFooterMarkup = ##f  evenFooterMarkup = ##f
}
\header { title = "Bang-chhun-hong - arrangement" tagline = ##f }
\score {
  <<
%(staves)s
  >>
  \layout { }
}
'''
_LY_STAFF = r'''    \new Staff \with { instrumentName = \markup { \small "%(name)s" } } {
      \clef %(clef)s \key c \major \time 4/4
      %(music)s \bar "|."
    }'''


def _build_lilypond(parsed: ParsedSolution) -> str:
    """Assemble the full .ly source for the arrangement (one staff per part)."""
    total16 = parsed.melody.total16
    carrier = set(parsed.carrier_part)
    by_part: dict[int, list[Note]] = {}
    for nt in parsed.notes:
        by_part.setdefault(nt.part, []).append(nt)
    staves = []
    # Engraving convention: the main melody on the TOP staff, then the rest high→low by
    # mean pitch. We force any carrier part(s) to the top (sorted among themselves by mean),
    # then the non-carrier parts high→low - so the tune is never buried mid-score.
    def mean_pitch(p):
        ns = by_part.get(p.index, [])
        return float(np.mean([n.pitch for n in ns])) if ns else 0.0
    carriers = sorted((p for p in parsed.parts if p.index in carrier),
                      key=mean_pitch, reverse=True)
    others = sorted((p for p in parsed.parts if p.index not in carrier),
                    key=mean_pitch, reverse=True)
    for p in carriers + others:
        ns = sorted(by_part.get(p.index, []), key=lambda n: (n.onset16, n.pitch))
        nm = p.name + (" *" if p.index in carrier else "")
        staves.append(_LY_STAFF % {"name": nm.replace('"', ""),
                                    "clef": _clef_for(ns, p.program),
                                    "music": _ly_part_music(ns, total16)})
    return _LY_TEMPLATE % {"staves": "\n".join(staves)}


def render_engraved_score_png(scored: Scored, melody: Melody | None = None) -> bytes | None:
    """Engrave the arrangement as proper staff notation (总谱) → PNG bytes. Primary path:
    hand-written LilyPond → `lilypond --png` (deterministic). Fallback: music21 → MusicXML
    → verovio → SVG → rsvg-convert. None on any failure (fail-soft - scoring never depends
    on this)."""
    parsed = scored.parsed
    # 1) LilyPond (primary)
    try:
        import shutil
        import subprocess
        import tempfile
        if shutil.which("lilypond"):
            ly = _build_lilypond(parsed)
            with tempfile.TemporaryDirectory(prefix="orch_ly_") as td:
                lyf = Path(td) / "s.ly"
                base = Path(td) / "out"
                lyf.write_text(ly)
                env = {**os.environ, "DISPLAY": ""}
                r = subprocess.run(
                    ["lilypond", "--png", "-dresolution=150", "--loglevel=ERROR",
                     "-o", str(base), str(lyf)],
                    capture_output=True, text=True, timeout=90, env=env)
                png = Path(str(base) + ".png")
                if png.is_file() and png.stat().st_size > 0:
                    return png.read_bytes()
    except Exception:
        pass
    # 2) verovio + rsvg-convert (fallback)
    try:
        return _engrave_verovio(parsed)
    except Exception:
        return None


def _engrave_verovio(parsed: ParsedSolution) -> bytes | None:
    """Fallback engraver: build a music21 Score → MusicXML → verovio SVG → rsvg PNG.
    Deterministic via verovio's xmlIdChecksum option."""
    try:
        import shutil
        import subprocess
        import tempfile
        import verovio
        import music21 as m21
    except Exception:
        return None
    sc = _build_music21(parsed)
    if sc is None:
        return None
    try:
        xml = m21.musicxml.m21ToXml.GeneralObjectExporter(sc).parse().decode("utf-8")
    except Exception:
        return None
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2400, "scale": 38, "adjustPageHeight": True,
                   "header": "none", "footer": "none", "xmlIdChecksum": True})
    if not tk.loadData(xml):
        return None
    svg = tk.renderToSVG(1)
    with tempfile.TemporaryDirectory(prefix="orch_vrv_") as td:
        svgf = Path(td) / "s.svg"
        pngf = Path(td) / "s.png"
        svgf.write_text(svg)
        if shutil.which("rsvg-convert"):
            subprocess.run(["rsvg-convert", "-w", "1600", str(svgf), "-o", str(pngf)],
                           check=True, timeout=30)
        else:
            try:
                import cairosvg
                cairosvg.svg2png(url=str(svgf), write_to=str(pngf), output_width=1600)
            except Exception:
                return None
        if pngf.is_file() and pngf.stat().st_size > 0:
            return pngf.read_bytes()
    return None


def _build_music21(parsed: ParsedSolution):
    """Build a music21 Score (one part per arrangement part) for the verovio fallback.
    None on failure."""
    try:
        import music21 as m21
    except Exception:
        return None
    try:
        spq = parsed.melody.grid_per_beat * 2          # sixteenths per quarter = 4
        score = m21.stream.Score()
        carrier = set(parsed.carrier_part)
        by_part: dict[int, list[Note]] = {}
        for nt in parsed.notes:
            by_part.setdefault(nt.part, []).append(nt)
        for p in parsed.parts:
            part = m21.stream.Part()
            events: dict[tuple[int, int], list[int]] = {}
            for nt in by_part.get(p.index, []):
                events.setdefault((nt.onset16, nt.dur16), []).append(nt.pitch)
            for (on, dur) in sorted(events):
                ql = dur / spq
                ps = sorted(events[(on, dur)])
                if len(ps) == 1:
                    el = m21.note.Note(ps[0]); el.quarterLength = ql
                else:
                    el = m21.chord.Chord(ps); el.quarterLength = ql
                part.insert(on / spq, el)
            part.partName = p.name + (" *" if p.index in carrier else "")
            score.insert(0, part)
        return score
    except Exception:
        return None
