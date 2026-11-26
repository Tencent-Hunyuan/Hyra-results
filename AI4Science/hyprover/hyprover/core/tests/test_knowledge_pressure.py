"""Bounded synthetic pressure scenarios; printed timings are observations, not SLOs."""
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import queue
import statistics
import threading
import time

from hyprover.core import KnowledgeStore, knowledge_markdown as md
from hyprover.core.schema import KnowledgeEntry, KnowledgeSnapshot


def _setting(name, default, maximum):
    value = int(os.environ.get(name, default))
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _latencies(values):
    ordered = sorted(values)
    return {"count": len(values), "p50_ms": round(statistics.median(ordered) * 1000, 3),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * .95))] * 1000, 3),
            "max_ms": round(max(ordered) * 1000, 3)}


def _writer(root, number, count, start, results):
    try:
        if not start.wait(20):
            raise AssertionError("writer start timeout")
        store = KnowledgeStore(Path(root))
        timings = []
        for index in range(count):
            kind = "experience" if index % 2 else "report"
            payload = dict(kind=kind, title=f"Writer {number} item {index}",
                           body="Synthetic keyword and explicit scope; no research performed.",
                           author=f"worker-{number}", role="worker", details=f"Synthetic detail {number}/{index}",
                           request_id=f"write-{index}", attributes={"subtype": "method"} if kind == "experience" else {})
            before = time.perf_counter()
            entry = store.record(**payload)
            timings.append(time.perf_counter() - before)
            if index % 7 == 0:
                before = time.perf_counter()
                assert store.record(**payload).entry_id == entry.entry_id
                timings.append(time.perf_counter() - before)
            if index % 5 == 0:
                before = time.perf_counter()
                store.record("report", "Shared exact request", "One shared report for each key.", "shared", "worker",
                             details=f"Shared detail {index}", request_id=f"shared-{index}")
                timings.append(time.perf_counter() - before)
        results.put(("writer", timings))
    except BaseException as exc:
        results.put(("error", f"writer {number}: {type(exc).__name__}: {exc}"))
        raise


def _reader(root, start, done, results):
    try:
        if not start.wait(20):
            raise AssertionError("reader start timeout")
        store = KnowledgeStore(Path(root))
        timings, revisions = [], []
        deadline = time.monotonic() + 120
        while not done.is_set() or len(timings) < 25:
            if time.monotonic() > deadline:
                raise AssertionError("concurrent readers exceeded deadline")
            before = time.perf_counter()
            snapshot = store.snapshot()
            page = store.search_page("keyword", kinds=["report", "experience"], limit=30, snapshot=snapshot)
            assert page["snapshot_revision"] == snapshot.revision
            assert len({row["id"] for row in page["results"]}) == len(page["results"])
            assert all(not row["is_fact"] for row in page["results"])
            if len(timings) % 10 == 0:
                assert md.parse(md.serialize(snapshot)) == snapshot
            revisions.append(snapshot.revision)
            timings.append(time.perf_counter() - before)
            time.sleep(.005)
        assert revisions == sorted(revisions)
        results.put(("reader", timings))
    except BaseException as exc:
        results.put(("error", f"reader: {type(exc).__name__}: {exc}"))
        raise


def test_multi_process_mixed_writes_retries_and_readers(tmp_path):
    workers = _setting("HYPROVER_STRESS_WORKERS", 8, 16)
    count = _setting("HYPROVER_STRESS_PER_WORKER", 25, 100)
    root = tmp_path / "pressure"
    store = KnowledgeStore.initialize(root, "synthetic-pressure")
    context = multiprocessing.get_context("spawn")
    start, done, results = context.Event(), context.Event(), context.Queue()
    processes = [context.Process(target=_writer, args=(str(root), i, count, start, results)) for i in range(workers)]
    processes += [context.Process(target=_reader, args=(str(root), start, done, results)) for _ in range(2)]
    writes, reads, finished = [], [], 0
    began = time.perf_counter()
    deadline = time.monotonic() + 120
    try:
        for process in processes:
            process.start()
        start.set()
        for _ in processes:
            try:
                role, payload = results.get(timeout=max(.1, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise AssertionError("pressure processes exceeded 120 seconds") from exc
            assert role != "error", payload
            if role == "writer":
                writes.extend(payload)
                finished += 1
                if finished == workers:
                    done.set()
            else:
                reads.extend(payload)
        for process in processes:
            process.join(5)
            assert not process.is_alive() and process.exitcode == 0
    finally:
        done.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        results.close()
        results.join_thread()
    elapsed = time.perf_counter() - began
    snapshot = store.snapshot()
    expected = workers * count + len(range(0, count, 5))
    assert snapshot.revision == expected and len(snapshot.entries) == expected
    assert not snapshot.facts
    assert len({e.display_id for e in snapshot.entries.values()}) == expected
    assert len(list((root / "knowledge/notes").glob("*.md"))) == expected
    assert md.parse(store.path.read_text()) == snapshot
    for entry in snapshot.entries.values():
        assert store.entry_detail(entry.entry_id)["details"]
    assert all(row["kind"] != "report" for row in store.search("", limit=1000))
    print("STRESS_METRICS " + json.dumps({"scenario": "multi_process_disk", "writers": workers, "readers": 2,
        "unique_records": expected, "write_calls": len(writes), "elapsed_s": round(elapsed, 3),
        "write_calls_per_s": round(len(writes) / elapsed, 3), "notebook_bytes": store.path.stat().st_size,
        "write_latency": _latencies(writes), "read_latency": _latencies(reads)}, sort_keys=True))


def test_contended_updates_have_one_winner_per_revision(tmp_path):
    store = KnowledgeStore.initialize(tmp_path / "cas", "synthetic-cas")
    entry = store.record("report", "Shared report", "Initial", "writer", "worker")
    rounds, contenders, conflicts = 10, 8, 0
    timings = []
    for round_number in range(rounds):
        revision = store.snapshot().entries[entry.entry_id].revision
        barrier = threading.Barrier(contenders, timeout=15)

        def update(index):
            barrier.wait()
            before = time.perf_counter()
            try:
                result = KnowledgeStore(store.root).record("report", entry.title, f"Round {round_number}, candidate {index}",
                    "writer", "worker", entry_id=entry.entry_id, expected_revision=revision,
                    details=f"Exact winning detail {round_number}/{index}")
                return result, time.perf_counter() - before
            except ValueError as exc:
                assert str(exc) == "entry_revision_conflict"
                return None, time.perf_counter() - before

        with ThreadPoolExecutor(max_workers=contenders) as executor:
            results = list(executor.map(update, range(contenders)))
        winners = [entry for entry, _ in results if entry is not None]
        assert len(winners) == 1 and winners[0].revision == revision + 1
        assert store.snapshot().entries[entry.entry_id] == winners[0]
        conflicts += sum(entry is None for entry, _ in results)
        timings.extend(duration for _, duration in results)
    assert store.snapshot().entries[entry.entry_id].revision == rounds + 1
    assert conflicts == rounds * (contenders - 1)
    assert len(list((store.root / "knowledge/notes").glob("*.md"))) == rounds
    print("STRESS_METRICS " + json.dumps({"scenario": "cas_updates", "contenders": contenders,
        "rounds": rounds, "commits": rounds, "expected_conflicts": conflicts,
        "latency": _latencies(timings)}, sort_keys=True))


def test_large_snapshot_filtered_query_capacity(tmp_path):
    count = _setting("HYPROVER_STRESS_QUERY_RECORDS", 5000, 10000)
    store = KnowledgeStore.initialize(tmp_path / "queries", "synthetic-queries")
    entries = {}
    for i in range(count):
        kind = "experience" if i % 2 else "report"
        eid = f"entry-{i:016x}"
        attrs = {"research_policy_version": 1}
        if kind == "experience":
            attrs["subtype"] = "method"
        entries[eid] = KnowledgeEntry(eid, kind, f"Synthetic item {i}", "keyword scope and recovery details",
                                      f"worker-{i % 8}", status="open" if kind == "experience" else "recorded",
                                      attributes=attrs)
    snapshot = md.prepare_display(KnowledgeSnapshot("synthetic-queries", count, entries=entries))
    md.validate(snapshot)
    expected = sum(e.author == "worker-3" for e in entries.values())
    timings = []
    for _ in range(12):
        before = time.perf_counter()
        page = store.search_page("keyword", subtype="method", author="worker-3", limit=20, snapshot=snapshot)
        assert page["total"] == expected
        assert all(row["kind"] == "experience" and row["author"] == "worker-3" for row in page["results"])
        timings.append(time.perf_counter() - before)
    print("STRESS_METRICS " + json.dumps({"scenario": "in_memory_snapshot_query_not_write_throughput",
        "records": count, "matching_records": expected, "serialized_bytes": len(md.serialize(snapshot).encode()),
        "query_latency": _latencies(timings)}, sort_keys=True))
