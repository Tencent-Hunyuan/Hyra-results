# Research Notebook Core

`hyprover.core` manages the notebook format, revisions, concurrent writes, proof dependencies, and fact revocation. The main agent and workers access the notebook through MCP tools. The verification service publishes facts. [Architecture](../../docs/ARCHITECTURE.md) · [MCP and Python interfaces](INTERFACES.md)

<a id="browse-by-task"></a>

## Read a Result

1. Open `KNOWLEDGE.md` or call `KnowledgeStore(project_dir).snapshot()` for complete statements and fact IDs. Later writes leave an existing snapshot unchanged.
2. Use `snapshot.topological(targets)` to find the valid facts a target uses, in dependency order. Use `get(fact_id)` to look up an individual statement and record.
3. To review an argument, follow its proof link to an attachment in `knowledge/proofs/`. In Python, `read_fact(fact_id, snapshot=snapshot)` reads the proof and checks its content ID. Load extended notes on demand with `entry_detail(entry_id)`. [Example research notebook](../../examples/README.md)

## Source Files

| File | Responsibility |
| --- | --- |
| `schema.py` | Facts, entries, snapshots, notebook formats, and fact content IDs |
| `knowledge.py` | Snapshot queries, concurrent writes, fact publication, and review |
| `knowledge_markdown.py` | Markdown serialization, record boundaries, and integrity checks |
| `entry_policy.py` | Experience subtypes, audit baselines, and creation retries |
| `glossary_global.json` | Reference for common mathematical notation |
| `_util.py` | Atomic writes and timestamps |

The data model documents revisions, associations, proof dependencies, and the write process. [Data model](DATA_MODEL.md)

## Offline Tests

`python -m pytest hyprover/core/` covers reads, writes, and failure recovery. `tests/test_knowledge_pressure.py` exercises concurrent reads and writes, creation retries, revision conflicts, and large snapshots. Adjust the load with `HYPROVER_STRESS_WORKERS`, `HYPROVER_STRESS_PER_WORKER`, and `HYPROVER_STRESS_QUERY_RECORDS`, and use pytest's `-s` to see `STRESS_METRICS`. The write lock serializes commits to the same notebook. `tests/test_storage_recovery.py` simulates interrupted writes in temporary projects and checks recovery.
