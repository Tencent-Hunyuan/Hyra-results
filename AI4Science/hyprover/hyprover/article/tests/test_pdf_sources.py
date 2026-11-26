import hashlib
import io
import shutil
from pathlib import Path

import pytest

from hyprover.article import evidence, literature, pdf_sources, references, render
from hyprover.article.models import ArticleConfig, ArticleError
from .test_pipeline import fixture
from .test_reference_refresh import META


def extractor(monkeypatch, *, pages=2, encrypted=False, chunks=None, returncode=0):
    commands = []
    monkeypatch.setattr(pdf_sources.shutil, "which", lambda name: "/usr/bin/" + name)

    def run(command, budget, **kwargs):
        commands.append(command)
        work = Path(command[command.index("--bind") + 1])
        if "/usr/bin/pdfinfo" in command:
            kwargs["stdout"].write(("Pages: " + str(pages) + "\nEncrypted: " + ("yes" if encrypted else "no") + "\n").encode())
        else:
            text = "\f".join(chunks if chunks is not None else ["Theorem 1. A complete source paragraph."] * pages) + "\f"
            (work / "source.txt").write_text(text)
        return returncode

    monkeypatch.setattr(pdf_sources, "run_command", run)
    return commands


def test_pdf_source_has_raw_hash_and_exact_physical_page_markers(monkeypatch):
    commands = extractor(monkeypatch)
    raw = b"%PDF-1.7\nfixture-source"
    text = references.extract_text(raw, "application/pdf", 4096)
    assert hashlib.sha256(raw).hexdigest() in text
    assert "[PDF page 1]" in text and "[PDF page 2]" in text
    assert text.count("[PDF page ") == 2
    assert len(commands) == 2
    for command in commands:
        assert "--unshare-all" in command and "--clearenv" in command
        assert "--cpu=45" in command and not any("api.key" in part for part in command)
        assert "--share-net" not in command and "/data/workspace" not in command
    assert references.source_kind(text, "https://arxiv.org/pdf/2301.12345") == "unconfirmed"


def test_pdf_extraction_accepts_input_larger_than_the_former_default(monkeypatch):
    extractor(monkeypatch, pages=1)
    raw = b"%PDF-1.7\n" + b"p" * (2 * 1024 * 1024)

    text = references.extract_text(raw, "application/pdf", ArticleConfig().max_source_bytes)

    assert "[PDF page 1]" in text
    assert hashlib.sha256(raw).hexdigest() in text


def test_remote_pdf_streams_to_the_extractor_and_keeps_its_byte_hash(monkeypatch):
    extractor(monkeypatch, pages=1)
    raw = b"%PDF-1.7\n" + b"p" * (2 * 1024 * 1024)

    class Response(io.BytesIO):
        status = 200

        def read(self, count):
            assert 0 < count <= 65536
            return super().read(count)

        def getheader(self, name, default=None):
            return "application/octet-stream" if name == "Content-Type" else default

    class Connection:
        sock = None

        def request(self, method, path, **kwargs):
            assert (method, path) == ("GET", "/item/fixture.pdf")

        def getresponse(self):
            return Response(raw)

        def close(self):
            pass

    monkeypatch.setattr(references, "_public_addresses", lambda host: ["8.8.8.8"])
    monkeypatch.setattr(references, "_PinnedHTTPS", lambda *args: Connection())

    def no_raw_extraction(*args, **kwargs):
        pytest.fail("PDF was processed as an in-memory body")

    monkeypatch.setattr(pdf_sources, "extract", no_raw_extraction)
    text, final = references.SourceReader(ArticleConfig(), True).get_text("http://numdam.org/item/fixture.pdf")
    assert final == "https://numdam.org/item/fixture.pdf"
    assert "[PDF page 1]" in text and hashlib.sha256(raw).hexdigest() in text
    reader = references.SourceReader(ArticleConfig(max_source_bytes=2097152), True)
    with pytest.raises(ArticleError, match="source_too_large"):
        reader.get_text(final)


@pytest.mark.parametrize("filename", ["info.txt", "source.txt"])
def test_sandbox_outputs_cannot_redirect_parent_reads(monkeypatch, tmp_path, filename):
    extractor(monkeypatch)
    original_run = pdf_sources.run_command
    secret = tmp_path / "outside-fixture.txt"
    secret.write_text("Private fixture; never read as a PDF source")

    def attacked(command, budget, **kwargs):
        result = original_run(command, budget, **kwargs)
        if (filename == "info.txt" and "/usr/bin/pdfinfo" in command
                or filename == "source.txt" and "/usr/bin/pdftotext" in command):
            path = Path(command[command.index("--bind") + 1]) / filename
            path.unlink()
            path.symlink_to(secret)
        return result

    monkeypatch.setattr(pdf_sources, "run_command", attacked)
    with pytest.raises(ArticleError, match="unsafe_pdf_extraction_output"):
        pdf_sources.extract(b"%PDF-1.7\\nfixture", 4096)


@pytest.mark.parametrize("fault,error", [
    ("pages", "page_budget"), ("encrypted", "encrypted_pdf"), ("scan", "unreadable_pages"),
    ("mismatch", "unreadable_pages"), ("corrupt", "corrupted"), ("failure", "extraction_failed")])
def test_bad_pdf_sources_are_not_silently_treated_as_read(monkeypatch, fault, error):
    options = {}
    if fault == "pages":
        options["pages"] = 201
    elif fault == "encrypted":
        options["encrypted"] = True
    elif fault == "scan":
        options["chunks"] = ["", ""]
    elif fault == "mismatch":
        options["chunks"] = ["One nonempty paragraph but a missing second page."]
    elif fault == "corrupt":
        options["chunks"] = ["A damaged formula " + "\ufffd" * 100] * 2
    else:
        options["returncode"] = 1
    extractor(monkeypatch, **options)
    with pytest.raises(ArticleError, match=error):
        pdf_sources.extract(b"%PDF-1.7\nfixture", 4096)


def test_missing_sandbox_and_invalid_signatures_do_not_use_unsafe_fallback(monkeypatch):
    monkeypatch.setattr(pdf_sources.shutil, "which", lambda name: None)
    with pytest.raises(ArticleError, match="environment_unavailable"):
        pdf_sources.extract(b"%PDF-1.7\nfixture", 4096)
    with pytest.raises(ArticleError, match="invalid_pdf_signature"):
        references.extract_text(b"not a PDF", "application/pdf", 4096)
    with pytest.raises(ArticleError, match="too_large"):
        pdf_sources.extract(b"%PDF-1.7\nfixture", 4)


@pytest.mark.skipif(not all(shutil.which(name) for name in ("bwrap", "pdflatex", "prlimit", "pdfinfo", "pdftotext")),
                    reason="Optional isolated PDF runtime unavailable")
def test_real_generated_pdf_is_readable_but_not_automatically_verified(project, tmp_path):
    frozen, ledger, article = fixture(project)
    root = tmp_path / "source-pdf"
    root.mkdir()
    render.write(root, render.render(article, ledger))
    compiled = render.compile_pdf(root)
    assert compiled["compiled"], compiled
    raw = (root / "paper.pdf").read_bytes()
    text = references.extract_text(raw, "application/pdf", 2097152)
    assert "A fixture result" in text and "[PDF page 1]" in text
    assert hashlib.sha256(raw).hexdigest() in text
    rid = references.add_reference(ledger, META)
    sid = references.add_source(ledger, rid, text, origin="https://www.ams.org/fixture.pdf",
                                kind=references.source_kind(text, "https://www.ams.org/fixture.pdf"))
    assert ledger["sources"][sid]["kind"] == "unconfirmed" and not ledger["checks"]
    references.validate(ledger, frozen)


def test_failed_publisher_request_keeps_actionable_sanitized_reason(project):
    from .conftest import publish
    frozen = evidence.capture(project.root, [publish(project)])
    ledger = references.seed(frozen)
    rid = references.add_reference(ledger, {**META, "fulltext_url": "https://www.ams.org/fixture.pdf"})
    reader = references.SourceReader(ArticleConfig(), True)

    def fail(url):
        raise ArticleError("source_http_403")

    reader.get_text = fail
    literature.publisher_sources(ledger, rid, reader)
    record = next(iter(ledger["retrieval_requests"].values()))
    assert record["error"] == "source_http_403" and record["error_type"] == "ArticleError"
    assert references.source_error(OSError("private path or token")) == "source_network_or_file_error"
