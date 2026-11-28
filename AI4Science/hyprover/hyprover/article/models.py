"""Versioned article contracts and bounded, side-effect-free configuration."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping

from hyprover import codex
from hyprover.core._util import atomic_write_text

FORMAT = "hyprover-article-v1"
STRICT = "strict"
FORMAT_COVERAGE = "format-coverage"
REVIEW_MODES = frozenset({STRICT, FORMAT_COVERAGE})
AUTHOR = "hyprover"
TEMPLATE = "arxiv_2608_13536v2"
DEFAULT_SOURCE_TEXT_BYTES = 2 * 1024 * 1024
ARTICLE_DOCUMENT_MAX_BYTES = 32 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,159}\Z")
TERMINAL = frozenset({"ready", "needs_review", "failed", "stale", "cancelled"})


class ArticleError(ValueError):
    """An article request cannot safely proceed."""


class ResponseContractError(ArticleError):
    """A model response still violates its contract after bounded correction."""

    def __init__(self, message: str, *, failures: list[dict] | None = None):
        super().__init__(message)
        self.failures = list(failures or [])


class RequestBudgetExceeded(ArticleError):
    """The exact model request is too large to dispatch safely."""


class EvidenceChanged(ArticleError):
    """A frozen article no longer describes the current evidence."""


class Cancelled(ArticleError):
    """The operator cancelled the independent article job."""


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(p in ("..", ".") for p in rel.parts):
        raise ArticleError("unsafe_article_path")
    path = Path(root).absolute() / rel
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ArticleError("symlink_article_path")
    return path


def read_text(path: Path, max_bytes: int = ARTICLE_DOCUMENT_MAX_BYTES) -> str:
    path = safe_path(path.parent, path.name)
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ArticleError("article_input_too_large")
    return raw.decode("utf-8")


def _nonfinite(value: str):
    raise ArticleError("nonfinite_article_json")


def read_json(path: Path, max_bytes: int = ARTICLE_DOCUMENT_MAX_BYTES) -> dict:
    value = json.loads(read_text(path, max_bytes), parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ArticleError("article_document_must_be_object")
    return value


def json_document(value: dict, max_bytes: int = ARTICLE_DOCUMENT_MAX_BYTES) -> str:
    document = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if len(document.encode("utf-8")) > max_bytes:
        raise ArticleError("article_document_too_large_to_save")
    return document


def write_json(path: Path, value: dict, *, max_bytes: int = ARTICLE_DOCUMENT_MAX_BYTES) -> None:
    path = safe_path(path.parent, path.name)
    atomic_write_text(path, json_document(value, max_bytes))


def require_keys(value: object, required: set[str], optional: set[str] | None = None) -> dict:
    if (not isinstance(value, dict) or not required <= value.keys()
            or value.keys() - required - (optional or set())):
        raise ArticleError("invalid_article_fields: " + ",".join(sorted(required)))
    return value


def string_list(value: object, *, nonempty: bool = False) -> list[str]:
    if (not isinstance(value, list) or (nonempty and not value)
            or any(not isinstance(v, str) or not v.strip() for v in value)
            or len(set(value)) != len(value)):
        raise ArticleError("article_requires_distinct_strings")
    return value


def nonempty_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArticleError("article_requires_nonempty_text")
    return value


def boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ArticleError(field + "_must_be_boolean")
    return value


@dataclass(frozen=True)
class ArticleConfig:
    auto_generate: bool = True
    model: str = codex.DEFAULT_MODEL
    effort: str = codex.DEFAULT_EFFORT
    # Keep the public configuration's serialized shape stable for frozen jobs.
    api_backend: str = "codex"
    reasoning_mode: None = None
    chunk_chars: int = 16000
    max_input_chars: int = 180000
    max_output_chars: int = 160000
    max_output_tokens: int | None = None
    max_calls: int = 80
    timeout_seconds: int = 1800
    total_seconds: int = 21600
    max_queries: int = 12
    max_sources: int = 48
    max_source_bytes: int | None = None
    source_timeout: int = 60
    max_revisions: int = 1
    allowed_source_hosts: tuple[str, ...] = (
        "arxiv.org", "export.arxiv.org", "api.crossref.org", "doi.org",
        "www.ams.org", "projecteuclid.org", "link.springer.com",
        "www.cambridge.org", "academic.oup.com", "annals.math.princeton.edu",
        "numdam.org", "www.numdam.org",
    )

    def __post_init__(self):
        boolean(self.auto_generate, "article.auto_generate")
        for name in ("model", "effort"):
            nonempty_text(getattr(self, name))
        if self.api_backend != "codex":
            raise ArticleError("unsupported_article_api_backend: configure the standard Responses API")
        if self.reasoning_mode is not None:
            raise ArticleError("unsupported_article_reasoning_mode: configure article reasoning effort instead")
        bounds = {
            "chunk_chars": (256, 64000), "max_input_chars": (2048, 1000000),
            "max_output_chars": (1024, 1000000), "max_calls": (1, 500),
            "timeout_seconds": (1, 10800), "total_seconds": (1, 259200),
            "max_queries": (0, 100), "max_sources": (0, 200),
            "source_timeout": (1, 360),
            "max_revisions": (0, 3),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ArticleError("invalid_article_config: " + name)
        if self.max_output_tokens is not None and (
                type(self.max_output_tokens) is not int or self.max_output_tokens < 1024):
            raise ArticleError("invalid_article_config: max_output_tokens")
        if self.max_source_bytes is not None and (
                type(self.max_source_bytes) is not int or not 1024 <= self.max_source_bytes <= 8388608):
            raise ArticleError("invalid_article_config: max_source_bytes")
        if (not isinstance(self.allowed_source_hosts, (list, tuple))
                or any(not isinstance(h, str) or not re.fullmatch(r"[a-z0-9.-]+", h)
                       or "." not in h for h in self.allowed_source_hosts)):
            raise ArticleError("invalid_article_source_hosts")
        object.__setattr__(self, "allowed_source_hosts", tuple(sorted(set(self.allowed_source_hosts))))

    @property
    def local_source_max_bytes(self) -> int:
        return self.max_source_bytes or DEFAULT_SOURCE_TEXT_BYTES

    def to_dict(self) -> dict:
        result = asdict(self)
        result["allowed_source_hosts"] = list(self.allowed_source_hosts)
        return result

    @classmethod
    def from_dict(cls, value: dict) -> ArticleConfig:
        require_keys(value, set(), {f.name for f in fields(cls)})
        return cls(**value)

    @classmethod
    def resolve(cls, metadata: dict, environ: Mapping[str, str] | None = None) -> ArticleConfig:
        env = os.environ if environ is None else environ
        section = metadata.get("article", {})
        require_keys(section, set(), {f.name for f in fields(cls)})
        values = dict(section)
        auto = env.get("HYPROVER_ARTICLE_AUTO")
        if auto is not None:
            if auto not in ("0", "1"):
                raise ArticleError("HYPROVER_ARTICLE_AUTO_must_be_0_or_1")
            values["auto_generate"] = auto == "1"
        for name in ("model", "effort"):
            configured = env.get("HYPROVER_ARTICLE_" + name.upper())
            if configured:
                values[name] = configured
            elif name not in values:
                values[name] = next((env[key] for key in (
                    "HYPROVER_MAIN_" + name.upper(), "HYPROVER_CODEX_" + name.upper())
                    if env.get(key)), getattr(cls, name))
        integer_overrides = ("max_input_chars", "max_output_tokens", "timeout_seconds")
        for name in ("api_backend", "reasoning_mode", *integer_overrides):
            configured = env.get("HYPROVER_ARTICLE_" + name.upper())
            if configured is not None:
                if name in integer_overrides:
                    try:
                        configured = int(configured)
                    except ValueError:
                        raise ArticleError("invalid_article_config: " + name) from None
                values[name] = configured
        return cls.from_dict(values)


def article_auto_enabled(metadata: dict | None = None) -> bool:
    """Check the kill switch before opening evidence, model or network resources."""
    override = os.environ.get("HYPROVER_ARTICLE_AUTO")
    if override is not None:
        if override not in ("0", "1"):
            raise ArticleError("HYPROVER_ARTICLE_AUTO_must_be_0_or_1")
        return override == "1"
    section = (metadata or {}).get("article", {})
    if not isinstance(section, dict):
        raise ArticleError("article_config_must_be_object")
    return boolean(section.get("auto_generate", True), "article.auto_generate")
