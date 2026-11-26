"""Standalone article commands; importing the parser never starts generation."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from .models import ArticleError


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or "/" in value or path.is_dir():
        return path.absolute()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in (".", ".."):
        raise ArticleError("invalid_article_project")
    return Path(os.environ.get("HYPROVER_AGENTS_ROOT", str(Path.cwd() / "runtime/projects"))).absolute() / value


def parser(root: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if root is None:
        root = argparse.ArgumentParser(prog="hyprover article", description="Local articles from verified proofs; no publication.")
    commands = root.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate from a committed completion or selected facts")
    generate.add_argument("project")
    generate.add_argument("--fact", action="append", dest="facts", help="Verified fact ID; repeat for multiple targets")
    generate.add_argument("--result", default="main/result.json", help="Completion result path relative to the project")
    generate.add_argument("--completion-root", help="Explicit absolute directory containing the completion receipts")
    generate.add_argument("--sources", help="Source manifest relative to the project; local texts belong in literature/")
    generate.add_argument("--background", action="store_true", help="Start an independent background task")
    generate.add_argument("--compile", action="store_true", help="Opt in to safely isolated PDF compilation")
    status = commands.add_parser("status", help="Inspect tasks and recheck evidence freshness")
    status.add_argument("project")
    status.add_argument("--run-id")
    resume = commands.add_parser("resume", help="Resume the same frozen task without repeating completed calls")
    resume.add_argument("project")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--background", action="store_true")
    resume.add_argument("--compile", action="store_true")
    resume.add_argument("--upgrade-writing", action="store_true", help="Recompose as a concise research paper, retaining the old version")
    resume.add_argument("--refresh-references", action="store_true", help="Supplement references and minimal context without rewriting proofs")
    stop = commands.add_parser("stop", help="Cancel an article task without affecting research")
    stop.add_argument("project")
    stop.add_argument("--run-id", required=True)
    worker = commands.add_parser("_run", help="internal background worker; normally started automatically")
    worker.add_argument("project")
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--resume", action="store_true")
    worker.add_argument("--compile", action="store_true")
    worker.add_argument("--upgrade-writing", action="store_true")
    worker.add_argument("--refresh-references", action="store_true")
    worker.add_argument("--dispatch-id", help=argparse.SUPPRESS)
    compile_command = commands.add_parser("compile", help="Compile an existing manuscript without calling a model")
    compile_command.add_argument("project")
    compile_command.add_argument("--run-id", required=True)
    recheck = commands.add_parser("recheck", help="Recheck a saved manuscript's format and evidence coverage locally")
    recheck.add_argument("project")
    recheck.add_argument("--run-id", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


def run(args: argparse.Namespace) -> int:
    from . import runtime

    try:
        project = project_path(args.project)
        if args.command == "generate":
            root = runtime.prepare(project, args.facts, result_path=args.result,
                                   completion_root=args.completion_root, sources=args.sources)
            result = runtime.launch(root, compile_output=args.compile) if args.background else runtime.work(
                project, root.name, compile_output=args.compile)
        elif args.command == "resume":
            root = runtime.job_path(project, args.run_id)
            options = dict(resume=True, compile_output=args.compile, upgrade_writing=args.upgrade_writing,
                           refresh_references=args.refresh_references)
            result = runtime.launch(root, **options) if args.background else runtime.work(project, args.run_id, **options)
        elif args.command == "status":
            result = runtime.status(project, args.run_id) if args.run_id else {"jobs": runtime.list_jobs(project)}
        elif args.command == "stop":
            result = runtime.stop(project, args.run_id)
        elif args.command == "compile":
            result = runtime.compile_article(project, args.run_id)
        elif args.command == "recheck":
            result = runtime.recheck(project, args.run_id)
        else:
            result = runtime.work(project, args.run_id, resume=args.resume, wait_for_launch=True,
                                  compile_output=args.compile, upgrade_writing=args.upgrade_writing,
                                  refresh_references=args.refresh_references, dispatch_id=args.dispatch_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.command in ("status", "stop"):
            return 0
        if args.command == "compile":
            return 0 if result.get("compilation", {}).get("compiled") is True else 2
        if args.command == "recheck":
            return 0 if result.get("state") == "ready" else 2
        if result.get("compilation", {}).get("compiled") is False:
            return 2
        return 0 if result.get("state") in ("ready", "queued", "running") else 2
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"state": "failed", "error": str(exc), "error_type": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
