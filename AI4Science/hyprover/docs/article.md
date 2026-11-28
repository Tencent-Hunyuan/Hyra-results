# Generating Articles from Verified Results

Article generation turns published facts into an English mathematical manuscript. It produces LaTeX sources, references, review records, and an optional PDF. The job freezes the target facts and their full proof dependencies without modifying them, then stores the manuscript and proof revisions separately. LaTeX uses the template included in the repository, with `hyprover` as the author. The template metadata identifies the source article. [Template metadata](../hyprover/article/templates/arxiv_2608_13536v2.json)

## Generate a Manuscript

Manuscripts are written and reviewed through Responses sessions with no tools enabled. Set `CODEX_API_BASE_URL` and `HYPROVER_CODEX_API_KEY` in `config/codex.env`. Leave `/responses` off the base URL. Remote endpoints use HTTPS, and loopback endpoints may use HTTP. Use `CODEX_BACKEND=api` for the article process. [Model access](getting-started.md#2-choose-a-model-backend)

By default, article generation reads the project's `main/result.json` and validates the completion receipt and its target facts. Use `--result` to specify a project-relative path and `--completion-root` for the absolute path to the `completion-…` receipt directory. To select published facts directly, repeat `--fact` with their internal IDs. The job also validates their proof dependencies.

```bash
./bin/hyprover article generate demo --background
./bin/hyprover article generate demo --fact "$FIRST_FACT_ID" --fact "$SECOND_FACT_ID"
```

Replace `demo` with a project name or absolute path. `--background` starts a background job and returns a `run_id` for subsequent commands. `output_dir` points to the project's `articles/<run-id>/` directory.

If the research deployment uses account authentication, `CODEX_BACKEND=chatgpt` in `config/hyprover.env` overrides the value in `config/codex.env`. Load the deployment configuration first, then launch the article process through Python with an API backend override. Use the same approach when resuming the job. Research sessions keep their account configuration.

```bash
. ./scripts/env.sh
CODEX_BACKEND=api "$HYPROVER_PY" -m hyprover.article generate demo --background
```

## Set Budgets and Sources

Set these fields in the `article` object of `project.json`. The environment variables listed in the table take precedence.

| Fields | Defaults or purpose |
| --- | --- |
| `model`, `effort` | Inherit the main agent's settings by default. Override them with `HYPROVER_ARTICLE_MODEL` and `HYPROVER_ARTICLE_EFFORT` |
| `max_calls`, `total_seconds` | At most 80 calls and 21,600 seconds per job |
| `timeout_seconds`, `max_input_chars`, `max_output_tokens` | Defaults are 1,800 seconds per call and 180,000 input characters, with an optional output token limit. Override them with `HYPROVER_ARTICLE_TIMEOUT_SECONDS`, `HYPROVER_ARTICLE_MAX_INPUT_CHARS`, and `HYPROVER_ARTICLE_MAX_OUTPUT_TOKENS` |
| `max_queries`, `max_sources`, `source_timeout` | Defaults are 12 searches, 48 sources, and 60 seconds to read a source |
| `allowed_source_hosts` | HTTPS hosts from which the job can read original sources |

Article generation uses arXiv, Crossref, and permitted original sources to check reference identity, provenance, and applicability. `HYPROVER_LITERATURE_SEARCH=0` disables network retrieval and disallows `--sources`. Jobs created with retrieval disabled retain that setting. If the deployment disables retrieval after an existing job was created with retrieval enabled, start a new job under the new settings. [Research literature search](literature-search.md)

When retrieval is enabled, pass `--sources literature/sources.json` to the generation command to provide original source material. Replace `reference` with an ID or unique alias in the reference ledger. Each item takes either `path` or `url`. A `path` points to UTF-8 text or HTML under the project's `literature/` directory, while a `url` must use a permitted HTTPS host. The `version` field is optional.

```json
{
  "sources": [
    {
      "reference": "example-reference",
      "path": "literature/example-source.txt",
      "version": "2nd edition"
    }
  ]
}
```

The job saves the source manifest and digests of local files. Before resuming, it checks whether the sources have changed.

## Review and Resume

The pipeline organizes the frozen evidence and references before drafting and rendering LaTeX. Inputs larger than the per-call limit are processed in proof-paragraph batches. `source-map.json` maps manuscript content to the original evidence.

New jobs use `review_mode=format-coverage` to check formatting and structural coverage of the original evidence. Proof repairs have separate review records, and unresolved findings go into `proof-repair.json` and `audit.json`. The job's `status` response reports review and typesetting status.

```bash
./bin/hyprover article status demo
./bin/hyprover article status demo --run-id "$ARTICLE_RUN_ID"
./bin/hyprover article stop demo --run-id "$ARTICLE_RUN_ID"
./bin/hyprover article resume demo --run-id "$ARTICLE_RUN_ID" --background
./bin/hyprover article recheck demo --run-id "$ARTICLE_RUN_ID"
```

| State | Meaning or action |
| --- | --- |
| `pending`, `queued`, `running` | The job has been created, queued, or started. See `phase` for the current stage |
| `ready` | The job passed the checks for the configured `review_mode`. See `audit.json` for scope and findings. Review the mathematical arguments and references manually before using the manuscript externally |
| `needs_review` | Check `audit.json` and, if present, `proof-repair.json` for unresolved proof, reference, or typesetting issues |
| `failed`, `cancelled`, `interrupted` | Check the error, cancellation, or exit reason, then resume with the original ID |
| `stale` | Use the error to inspect the evidence, template, or outputs. Usually, start a new job from current evidence |

`stop` requests cancellation, and `resume` continues the same job from its checkpoint. Use `recheck` to rerun local formatting and evidence coverage checks on an existing manuscript and save the findings to the job's review records. `resume --refresh-references` supplements references within the remaining budget. Finish any pending proof revisions before requesting it. `resume --upgrade-writing` reworks the manuscript body, retains the previous version, and runs review again.

Jobs are bound to fingerprints of their article configuration and model endpoint. A resumed job uses the original evidence, endpoint, and budget. Credentials may be rotated. To migrate a job, copy the project and its `articles/` directory. Copy any `completion-*` receipt directories stored outside the project to the project's `verify-runs/`, then use the original job ID. If you see `article_template_changed`, resume with the original software version or start a new job with the current version and valid evidence.

## Compile a PDF

The job directory contains `paper.tex`, `references.tex`, and `references.bib`. Use `generate --compile` or `resume --compile` to include PDF compilation, or compile the saved manuscript separately. `status`, `stop`, `recheck`, and standalone `compile` run locally.

```bash
./bin/hyprover article compile demo --run-id "$ARTICLE_RUN_ID"
```

Isolated compilation requires Linux with user namespace support. Text extraction from external PDF references also requires Poppler. The following command installs the same tools used in Ubuntu CI. The saved LaTeX remains usable if the isolated tools are missing. The job records compilation of `paper.pdf` separately from manuscript review.

```bash
sudo apt-get install --no-install-recommends bubblewrap util-linux poppler-utils \
  texlive-latex-base texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended texlive-fonts-extra lmodern
```

## Batch Processing and Local Delivery

A collection directory contains the completed projects and `export-manifest.json`. Under `projects`, use a unique `benchmark` identifier for each member and give its path relative to the collection directory in `directory`. Jobs are processed in `benchmark` order, with the notebook and completion receipt read for each project.

```json
{
  "projects": [
    {"benchmark": "demo", "directory": "demo"}
  ]
}
```

```bash
. ./scripts/env.sh
python -m hyprover.article.collection /path/to/collection --jobs 2
python -m hyprover.article.collection /path/to/collection --resume
python -m hyprover.article.delivery /path/to/collection
```

`--jobs` accepts 1 to 9 concurrent jobs and defaults to 2. `article-generation.json` records progress and job IDs. Add `--compile` to attempt PDF compilation and delivery. `delivery` summarizes review status and writes `paper/` for members eligible for export. If research uses account authentication, load the deployment configuration and set `CODEX_BACKEND=api` for the `collection` command, as for an individual article job.

Retrieve an individual manuscript from `articles/<run-id>/`, or call `hyprover.article.delivery.export(project, run_id)` to deliver it to `paper/`, with `project` as a `pathlib.Path`. Follow the version pointer and verification records referenced by `paper/current.json` to retrieve matching manuscript, reference, and review files.
