"""Bounded, network-isolated PDF-to-text extraction with immutable page provenance."""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from itertools import chain
from pathlib import Path

from .budget import Budget, run_command
from .models import ArticleError

MAX_PAGES = 200
MAX_TEXT_BYTES = 8 * 1024 * 1024


def extract(raw: bytes, max_bytes: int | None = None, *, timeout: int = 60,
            deadline: float | None = None, cancelled=None) -> str:
    if max_bytes is not None and len(raw) > max_bytes:
        raise ArticleError("pdf_source_too_large")
    budget = Budget(timeout, "pdf_extraction_unavailable_or_timed_out", deadline=deadline, cancelled=cancelled)
    return extract_chunks((raw,), budget=budget)


def extract_chunks(chunks: Iterable[bytes], *, timeout: int = 60, budget: Budget | None = None) -> str:
    """Stage a streamed PDF and retain its original byte hash."""
    if budget is None:
        budget = Budget(timeout, "pdf_extraction_unavailable_or_timed_out")
    budget.remaining()
    stream = iter(chunks)
    first = next(stream, b"")
    if not first.startswith(b"%PDF-"):
        raise ArticleError("invalid_pdf_signature")
    tools = {name: shutil.which(name) for name in ("bwrap", "prlimit", "pdfinfo", "pdftotext")}
    if not all(tools.values()):
        raise ArticleError("safe_pdf_extraction_environment_unavailable")
    with tempfile.TemporaryDirectory(prefix="hyprover-source-pdf-") as temporary:
        work = Path(temporary)
        sha = hashlib.sha256()
        with (work / "source.pdf").open("wb") as staged:
            for block in chain((first,), stream):
                budget.remaining()
                staged.write(block)
                sha.update(block)
        command = [tools["prlimit"], "--cpu=45", "--as=536870912", "--fsize=" + str(MAX_TEXT_BYTES),
                   "--nproc=32", "--nofile=64", tools["bwrap"], "--unshare-all", "--die-with-parent",
                   "--new-session", "--clearenv"]
        for location in ("/usr", "/lib", "/lib64"):
            if Path(location).exists():
                command += ["--ro-bind", location, location]
        command += ["--tmpfs", "/tmp", "--dir", "/proc", "--dev", "/dev", "--bind", str(work), "/work",
                    "--chdir", "/work", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin"]

        def run(arguments, output):
            try:
                with output.open("wb") as handle:
                    status = run_command(command + arguments, budget, stdout=handle, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ArticleError("pdf_extraction_unavailable_or_timed_out") from exc
            if status:
                raise ArticleError("pdf_extraction_failed_or_encrypted")

        info_path = work / "info.txt"
        run([tools["pdfinfo"], "source.pdf"], info_path)
        if info_path.is_symlink() or not info_path.is_file():
            raise ArticleError("unsafe_pdf_extraction_output")
        if info_path.stat().st_size > 32768:
            raise ArticleError("pdf_metadata_exceeds_budget")
        info = info_path.read_text(encoding="utf-8", errors="replace")
        pages = re.search(r"^Pages:\s+(\d+)\s*$", info, re.M)
        if not pages or not 1 <= int(pages.group(1)) <= MAX_PAGES:
            raise ArticleError("pdf_page_budget_exceeded_or_invalid")
        if re.search(r"^Encrypted:\s+yes", info, re.M):
            raise ArticleError("encrypted_pdf_requires_accessible_source")
        run([tools["pdftotext"], "-layout", "-enc", "UTF-8", "source.pdf", "source.txt"], work / "extract.log")
        text_path = work / "source.txt"
        if text_path.is_symlink():
            raise ArticleError("unsafe_pdf_extraction_output")
        if not text_path.is_file() or text_path.stat().st_size > MAX_TEXT_BYTES:
            raise ArticleError("pdf_extracted_text_exceeds_budget")
        try:
            text = text_path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise ArticleError("pdf_extracted_text_encoding_invalid") from exc
        chunks = text.split("\f")
        if chunks and not chunks[-1].strip():
            chunks.pop()
        if len(chunks) != int(pages.group(1)) or any(len(chunk.strip()) < 20 for chunk in chunks):
            raise ArticleError("pdf_has_unreadable_pages_requires_ocr_or_alternate_source")
        if "\x00" in text or text.count("\ufffd") > max(3, len(text) // 1000):
            raise ArticleError("pdf_text_corrupted_requires_alternate_source")
        provenance = "PDF SOURCE SHA256: " + sha.hexdigest() + "\nPDF PAGES: " + str(len(chunks))
        provenance += "\nPDF text extraction is not mathematical verification; ambiguous formula glyphs require another source.\n"
        budget.remaining()
        return provenance + "\n".join("\n[PDF page " + str(index) + "]\n" + chunk.rstrip()
                                       for index, chunk in enumerate(chunks, 1)) + "\n"
