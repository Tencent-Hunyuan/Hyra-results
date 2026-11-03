# Checkpoint integrity, durability, and crash recovery

A checkpointed search stops on its first logical witness. The finding piece
gets a terminal `F` record and **never gets a `C` completion record**. The
same `consume()` mutex protects witness adoption, `F` persistence, and every
completion append. Once a witness has been adopted, later consumers return
without appending anything. The parser rejects complete records after `F`.

The header and every C/F record are written as an entire line, followed by
a checked stdio flush and
`fsync(fileno(file))`, retrying `EINTR`. Each completion chunk has one sync
(regardless of task count). Append returns only after successful sync; any write, flush, or sync error exits nonzero before a find-bearing
progress or done result. A resumed complete `F` is also synced before its
witness is re-emitted, because the earlier process may have died between
flush and sync. Resumed C records are synced for the same reason. `fclose`
is checked before `done`; a close error exits 1 without a final result. A
find-bearing progress event may already have been emitted after a successful
F sync, but the Python wrapper requires exit 0 and `done` to accept a result.
On checkpoint I/O failure, the process flushes stderr and calls `_Exit(1)`: ordinary
`exit` can run CUDA teardown handlers while another GPU slot is active. GPU short-write
injection reproduced that race (SIGSEGV instead of exit 1). Buffered checkpoint bytes
are uncommitted and may be lost on this error path; restart replays their work. Generation and refinement finds use the same persistence
method; coverage certificates continue to refuse checkpoints.

If the process dies before a complete `F` exists, the finding piece has no
completion record and is searched again at the original bound. A final line
without its newline is discarded. This includes a torn `F` whose numerical
fields happen to be complete. Other workers may observe the lowered cap and
finish first, but they cannot complete the finding piece. If a complete `F`
exists, resume validates its logical witness and re-emits it. These two cases
prevent a process crash from converting a find into a clean no-find result.
The exact instance, cap, policy, and executable remain bound by the header.

Version 2 seeds zlib's IEEE CRC-32 with the exact header and newline. Every
C/F line adds ` crc32=<8 lowercase hex digits>`; its CRC is the previous CRC
updated with the payload and newline, excluding the checksum suffix. C includes
its round number; there are no standalone round records. Every complete record
is checked in order, then all task bookkeeping and any witness are validated,
before truncating a torn tail or appending. A missing/wrong CRC exits 6 with
the file untouched. A syntactically valid aborted index is covered by this
check. CRC protects against accidental corruption, not adversarial edits or
recomputed checksums; it is not a cryptographic hash. A lost suffix remains
safe because unfinished work is replayed. Version-1 files lack this chain and
are explicitly refused, requiring a new checkpoint file.

A file consisting solely of a strict prefix of this run's expected header
line restarts fresh (`resumed=false`), including zero bytes and a header missing
only its newline. Any other header mismatch still exits 6 without changing
the file. Partial header recovery uses the same separate lock as body recovery.

## Tests

`test_checkpoint_crash.py` compiles the actual native driver with an explicit
test-only preinclude, `checkpoint_crash_hooks.h`. Ordinary package builds do
not include it; the native hook macros expand to no-ops. Each recovery uses
the **same executable and binary hash** as the killed run, with only its
runtime fault selector removed. A header mismatch cannot make a test pass.

The preserved [[18,1,3]] fixture has the verified logical `Z15 X16 Z17`. At
split1/file root order it occurs in task46, after earlier pieces have already
written completion records. The tests require its task to remain absent from
every completion interval, before and after recovery. Every resumed run must
successfully report this witness; merely refusing the file does not pass.

* Seventeen kill boundaries span core detection/capture, CPU cap publication,
  witness handoff, locked consumption, write/flush/sync, and reporting. Each
  is tested with threads1/chunk1 and threads4/chunk8.
* Every byte offset of the final `F` line is explicitly written and flushed
  before `SIGKILL`, including zero bytes and the complete newline. Recovery
  must reconstruct the exact uninterrupted checkpoint bytes.
* Eight cases kill during initial generation and budget-driven refinement.
* Short writes, write `ENOSPC`, flush `EIO`/`ENOSPC`, sync `EIO`/`ENOSPC`,
  and close `EIO`/`ENOSPC` are injected for C and F. Each must exit 1 with
  no `done`; resumed no-find counts or independently checked witnesses must
  match the baseline. C faults are also tested before a later find.
* Every byte prefix of the header is explicitly written/flushed before kill;
  recovery must reproduce the uninterrupted bytes. Each nonmatching prefix
  is rejected untouched. Every byte offset in a C record is similarly killed
  and resumed, including budget-driven aborted-task refinement.
* Six C write/flush/sync kill boundaries retain exact no-find counts after
  recovery. The sync counter must match the C chunk count. Both C and F
  retry `EINTR`; restored records cannot bypass sync or close failures.
* The integrity suite flips an aborted index digit while retaining valid range,
  count, and sort order; it also changes counts, checksums and witness labels,
  deletes/swaps/duplicates records, and checks refusal of version 1. Recomputed
  CRCs in separate parser tests ensure semantic validation still rejects
  malformed task bookkeeping.
* The manual GPU test covers the device-result/CPU-witness handoff and all
  seventeen shared host boundaries with two GPU slots. GPU execution is
  separate evidence; CPU results do not establish GPU crash coverage. GPU C/F
  I/O tests apply the same eight error types and resume the same executable.

Run CPU tests with:

```sh
PYTHONPATH=src:tests python3 -m pytest tests/native/test_checkpoint.py tests/native/test_checkpoint_crash.py -m 'not gpu' -q
```

Set `HYQEC_CHECKPOINT_AUDIT_DIR` to retain every command, injected fault,
exit status, stdout, stderr, and checkpoint snapshot. Run the manual GPU job
only on an allocated device:

```sh
CUDA_VISIBLE_DEVICES=0 HYQEC_TEST_GPU=1 PYTHONPATH=src:tests \
  python3 -m pytest tests/native/test_checkpoint_crash.py -m gpu -q
```

B&B and the fault-injection builds require zlib development headers/library
(`-lz`).

These are real process-kill and injected I/O-failure tests, not hardware
power-loss tests. Durability relies on the filesystem honoring successful
`fsync`. Checkpoints remain trusted local progress records, not independently
verifiable coverage proofs. Version 1 synced F but only flushed C and ignored close errors; no false lower
bound from the C durability omission alone was reproduced before this fix.
