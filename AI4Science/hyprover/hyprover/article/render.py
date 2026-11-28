"""Deterministic AMS-style rendering; generated TeX is not executable input."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path

from hyprover.core._util import atomic_write_text

from .budget import Budget, OperationTimeout, run_command
from .models import (AUTHOR, TEMPLATE, ArticleError, read_json, read_text, safe_path, text_hash, write_json)
from .references import identifiers, normal, usable_check

_MARKER = re.compile(r"\[\[(cite|ref):([A-Za-z][A-Za-z0-9_.:-]{0,159})\]\]")
_COMMAND = re.compile(r"\\([A-Za-z]+|[^A-Za-z])")
_COMMANDS = set("""
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
mathbb mathcal mathscr mathfrak mathrm mathit mathsf mathtt mathbf boldsymbol bm mathds mathbbm
text textnormal textrm textit textbf textsf texttt emph operatorname operatornamewithlimits
frac dfrac tfrac binom dbinom tbinom sqrt overline underline widehat widetilde hat tilde bar vec dot ddot
sum prod coprod int iint iiint oint bigcup bigcap bigvee bigwedge lim limsup liminf inf sup min max det dim ker hom gcd deg log ln exp sin cos tan arcsin arccos arctan sinh cosh tanh
in notin ni subset supset subseteq supseteq subsetneq supsetneq nsubseteq nsupseteq cup cap setminus smallsetminus emptyset varnothing forall exists nexists neg land lor lnot
le leq leqslant ge geq geqslant ne neq equiv sim simeq approx asymp cong propto ll gg prec succ preceq succeq
rightarrow leftarrow leftrightarrow Rightarrow Leftarrow Leftrightarrow longrightarrow longleftarrow longleftrightarrow Longrightarrow Longleftarrow Longleftrightarrow mapsto longmapsto hookrightarrow hookleftarrow to gets implies iff
infty partial nabla ell hbar Re Im top bot perp parallel nparallel mid nmid vert Vert lvert rvert lVert rVert langle rangle lceil rceil lfloor rfloor lbrace rbrace
cdot cdots ldots vdots ddots dots dotsc dotsb dotsm dotsi dotso colon mathop Pr
lVert rVert nleq ngeq lesssim gtrsim coth csc sec cot
cdotp times div pm mp otimes oplus odot circ bullet star ast wedge vee tensor
left right middle big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr
underbrace overbrace underset overset stackrel substack limits nolimits
quad qquad enspace thinspace medspace thickspace hspace vspace phantom hphantom vphantom smash
begin end item displaystyle textstyle scriptstyle scriptscriptstyle pmod bmod mod pod
nonumber notag intertext shortintertext tagnotused
R Sph Ball cP cE cA Aut vol degree supp Hom End GL OO Lip Int Span Sym Gr dd ip
""".split())
_ADDITIONAL_MATH_COMMANDS = {"arg", "Subset", "not", "rightharpoonup", "diamond",
                             "uparrow", "downarrow", "updownarrow", "Uparrow", "Downarrow", "Updownarrow"}
_SPECIAL = set("()[]{}|,;:! %&#_$\\ ") | {"\n", "\r", "'", '"', "`", "^", "~", "="}
_ENVIRONMENTS = {"aligned", "alignedat", "gathered", "split", "cases", "matrix", "pmatrix", "bmatrix",
                 "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "array", "enumerate", "itemize",
                 "equation*", "align*", "gather*"}
_FORBIDDEN_PROSE = re.compile(r"declaration on the use of ai|conflict(?:s)? of interest|data availability|"
                             r"funding statement|funded by|supported by (?:the )?(?:grant|fund)|"
                             r"acknowledg(?:e)?ments|ai[- ]use declaration", re.I)
_NUMBERED = {"theorem", "lemma", "fact", "corollary", "problem", "conjecture", "definition", "example", "remark", "equation"}


def template() -> tuple[str, dict]:
    root = files("hyprover.article").joinpath("templates")
    text = root.joinpath(TEMPLATE + ".tex").read_text(encoding="utf-8")
    profile = json.loads(root.joinpath(TEMPLATE + ".json").read_text(encoding="utf-8"))
    if (text_hash(text) != profile["template_sha256"] or profile["author"] != AUTHOR
            or profile["id"] != TEMPLATE):
        raise ArticleError("article_template_changed")
    return text, profile


def escape(text: object) -> str:
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "%": r"\%", "&": r"\&",
                    "#": r"\#", "_": r"\_", "$": r"\$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(text))


def markers(text: str, kind: str) -> set[str]:
    return {identifier for found, identifier in _MARKER.findall(text) if found == kind}


def restore_json_escapes(text: str, source: str) -> tuple[str, list[dict]]:
    commands = set(_COMMAND.findall(source))
    changes = []
    def restore(match):
        command = {"\b": "b", "\f": "f"}[match.group(1)] + match.group(2)
        if command not in commands or command not in _COMMANDS:
            raise ArticleError("tex_escape_repair_not_grounded_in_source")
        changes.append({"offset": match.start(), "control_code": ord(match.group(1)), "command": command})
        return "\\" + command
    fixed = re.sub(r"([\x08\x0c])([A-Za-z]+)", restore, text)
    if any(ord(c) < 32 and c not in "\n\r\t" for c in fixed):
        raise ArticleError("unrecoverable_tex_control_character")
    return fixed, changes


def safe_tex(text: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ArticleError("empty_article_block")
    if any(ord(c) < 32 and c not in "\n\r\t" for c in text):
        raise ArticleError("unescaped_control_character_in_tex: escape LaTeX backslashes in JSON")
    if ("^^" in text or "@@HYPROVER_" in text or "\x00" in text
            or _FORBIDDEN_PROSE.search(text) or re.search(r"(?<!\\)[%#]", text)):
        raise ArticleError("unsafe_or_disallowed_article_text")
    if "[[" in _MARKER.sub("", text) or "]]" in _MARKER.sub("", text):
        raise ArticleError("invalid_article_reference_marker")
    for command in _COMMAND.findall(text):
        if command not in _COMMANDS and command not in _ADDITIONAL_MATH_COMMANDS and command not in _SPECIAL:
            raise ArticleError("unapproved_tex_command: " + command)
    stack = []
    for kind, name in re.findall(r"\\(begin|end)\s*\{([^}]+)\}", text):
        if name not in _ENVIRONMENTS:
            raise ArticleError("unapproved_tex_environment")
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            raise ArticleError("unbalanced_tex_environment")
    if stack:
        raise ArticleError("unbalanced_tex_environment")
    balance = 0
    for match in re.finditer(r"\\.|[{}]", text):
        token = match.group()
        if token == "{":
            balance += 1
        elif token == "}":
            balance -= 1
            if balance < 0:
                raise ArticleError("unbalanced_tex_braces")
    if balance:
        raise ArticleError("unbalanced_tex_braces")


def walk_blocks(article: dict):
    yield article["abstract"]
    for section in article["sections"]:
        for block in section["blocks"]:
            yield block
            yield from block.get("parts", [])


def render_text(text: str, ledger: dict, labels: set[str], aliases: dict | None = None) -> str:
    safe_tex(text)
    def replace(match):
        kind, identifier = match.groups()
        if kind == "ref":
            if identifier not in labels:
                raise ArticleError("dangling_article_cross_reference")
            return r"\ref{b:" + identifier + "}"
        if not usable_check(ledger, identifier):
            raise ArticleError("unverified_article_citation")
        check = ledger["checks"][identifier]
        rid = (aliases or {}).get(check["reference_id"], check["reference_id"])
        if check.get("evidence_level") in ("standard_reference", "background_reference"):
            return r"\cite{" + rid + "}"
        locator = escape(check["locator"]).replace("]", r"{]}").replace("[", r"{[}")
        return r"\cite[" + locator + "]{" + rid + "}"
    return _MARKER.sub(replace, text)


def bibliography_text(text: object) -> str:
    import unicodedata
    result = []
    for char in str(text):
        name = unicodedata.name(char, "")
        normalized = unicodedata.normalize("NFKC", char)
        if name.startswith(("MATHEMATICAL ", "DOUBLE-STRUCK ")) and len(normalized) == 1 and normalized.isascii() and normalized.isalnum():
            style = "\\mathbb{" + normalized + "}" if "DOUBLE-STRUCK" in name else "\\mathbf{" + normalized + "}" if "BOLD" in name else normalized
            result.append("\\ensuremath{" + style + "}")
        elif name.startswith(("SUPERSCRIPT ", "SUBSCRIPT ")) and normalized.isascii() and normalized.isalnum():
            result.append(("\\textsuperscript{" if name.startswith("SUPERSCRIPT") else "\\textsubscript{") + normalized + "}")
        elif char in "αβγδεθλμπσφωΔΘΛΠΣΦΩ":
            command = dict(zip("αβγδεθλμπσφωΔΘΛΠΣΦΩ", "alpha beta gamma delta epsilon theta lambda mu pi sigma phi omega Delta Theta Lambda Pi Sigma Phi Omega".split()))[char]
            result.append("\\ensuremath{\\" + command + "}")
        else:
            result.append(escape(char))
    return "".join(result)


def author_sort_key(authors) -> tuple:
    names = [authors] if isinstance(authors, str) else authors
    return tuple((normal(name.split(",", 1)[0] if "," in name else name.rsplit(" ", 1)[-1]), normal(name))
                 for name in names)


def bibliography(ledger: dict, used: set[str], aliases: dict | None = None) -> tuple[str, str]:
    rids = {(aliases or {}).get(ledger["checks"][cid]["reference_id"], ledger["checks"][cid]["reference_id"]) for cid in used}
    rows = [ledger["references"][rid] for rid in rids]
    rows.sort(key=lambda r: (author_sort_key(r["metadata"].get("authors", [])), normal(r["metadata"].get("title", "")), r["id"]))
    entries, bib = [], []
    for row in rows:
        if not row["metadata_verified"] or row["conflicts"]:
            raise ArticleError("unverified_bibliography_entry")
        data, rid = row["metadata"], row["id"]
        authors = data.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        title = bibliography_text(data["title"]) if aliases is not None else escape(data["title"])
        parts = [escape(", ".join(authors)), title]
        if data.get("edition"):
            parts.append(escape(data["edition"]) + " ed.")
        is_chapter = data.get("type") in ("book-chapter", "book-section")
        is_book = not is_chapter and ("book" in str(data.get("type", "")) or data.get("type") == "monograph" or bool(data.get("edition")))
        if is_book:
            parts.extend(escape(data[k]) for k in ("series", "publisher", "address") if data.get(k))
            parts.append(escape(data.get("year", "")))
        else:
            venue = escape(data.get("venue", ""))
            if data.get("volume"):
                venue += " " + escape(data["volume"])
            if data.get("number"):
                venue += "(" + escape(data["number"]) + ")"
            venue += " (" + escape(data.get("year", "")) + ")"
            if data.get("pages"):
                venue += " " + escape(data["pages"]).replace("–", "--")
            parts.append(venue.strip())
            if is_chapter:
                parts.extend(escape(data[k]) for k in ("series", "publisher") if data.get(k) and data[k] != data.get("venue"))
        doi, aid = identifiers(data)
        if doi:
            parts.append(r"\url{https://doi.org/" + escape(doi) + "}")
        elif aid:
            parts.append("arXiv:" + escape(aid))
        entries.append(r"\bibitem{" + rid + "} " + ", ".join(p for p in parts if p) + ".")
        entry_type = "incollection" if is_chapter else "book" if is_book else "misc" if aid and not doi else "article"
        values = {"author": " and ".join(authors), "title": data["title"], "year": data.get("year", "")}
        mapping = {"venue": "journal", "volume": "volume", "number": "number", "pages": "pages",
                   "publisher": "publisher", "edition": "edition", "address": "address", "doi": "doi", "arxiv": "eprint"}
        if is_chapter:
            mapping["venue"] = "booktitle"
            mapping["series"] = "series"
        values.update({dest: data[key] for key, dest in mapping.items() if data.get(key)})
        if aid:
            values["archivePrefix"] = "arXiv"
        bib.append("@" + entry_type + "{" + rid + ",\n" + ",\n".join(
            "  " + key + " = {" + (title if key == "title" else escape(value)) + "}" for key, value in values.items()) + "\n}")
    tex = "\\begin{thebibliography}{99}\n\\vskip 10pt\n\n" + "\n\n".join(entries) + "\n\n\\end{thebibliography}\n" if entries else ""
    return tex, "\n\n".join(bib) + ("\n" if bib else "")


def render(article: dict, ledger: dict) -> dict[str, str]:
    source, profile = template()
    blocks = list(walk_blocks(article))
    labels = {s["id"] for s in article["sections"]} | {
        b["id"] for b in blocks if b["kind"] in _NUMBERED or b["kind"] == "subsection"}
    used = set().union(*(markers(b["text"], "cite") for b in blocks))
    from . import credit
    editorial = article.get("writing_revision", 0) >= 2
    heading_break = "\\par\\nobreak\\noindent\n" if article.get("writing_revision", 0) >= 3 else "\\par\\nobreak\\vspace{\\baselineskip}\\noindent\n"
    aliases = credit.bibliography_aliases(ledger, used) if editorial else None
    by_id = {b["id"]: b for b in blocks}
    adjacent_proofs = {b["id"] for s in article["sections"] for i, b in enumerate(s["blocks"])
                       if b["kind"] == "proof" and i > 0 and s["blocks"][i - 1]["id"] == b["proves"]}
    body = []
    def block_tex(block):
        text = render_text(block["text"], ledger, labels, aliases)
        kind, identifier = block["kind"], block["id"]
        if kind == "paragraph":
            return text
        if kind == "subsection":
            heading = r"\subsection{" + escape(block["text"]) + "}\\label{b:" + identifier + "}"
            return heading + ("\\mbox{}" + heading_break if editorial else "")
        if kind == "step":
            heading = r"\medskip\noindent\textbf{" + escape(block.get("heading", "Step")) + ".}"
            return heading + (heading_break if editorial else " ") + text
        if kind == "proof":
            target = block["proves"]
            if target not in labels:
                raise ArticleError("proof_has_no_statement")
            proof = text + "\n\n" + "\n\n".join(block_tex(p) for p in block.get("parts", []))
            heading = "" if identifier in adjacent_proofs else "[Proof of " + by_id[target]["kind"].title() + "~\\ref{b:" + target + "}]"
            return "\\begin{proof}" + heading + "\n" + proof + "\n\\end{proof}"
        if kind in _NUMBERED:
            return "\\begin{" + kind + "}\\label{b:" + identifier + "}\n" + text + "\n\\end{" + kind + "}"
        raise ArticleError("unsupported_article_block")
    roles = [s["role"] for s in article["sections"]]
    if roles != profile["section_roles"]:
        raise ArticleError("article_section_style_mismatch")
    for section in article["sections"]:
        expected = profile["section_titles"].get(section["role"])
        if expected and section["title"] != expected:
            raise ArticleError("article_section_title_mismatch")
        if _FORBIDDEN_PROSE.search(section["title"]):
            raise ArticleError("disallowed_article_section")
        body.append(r"\section{\normalfont\bfseries " + escape(section["title"]) + "}\\label{b:" + section["id"] + "}")
        body.extend(block_tex(block) for block in section["blocks"])
    for code in article["msc"]:
        if not re.fullmatch(r"\d{2}[A-Z]\d{2}", code):
            raise ArticleError("invalid_msc_code")
    substitutions = {"TITLE": render_text(article["title"]["text"], ledger, labels),
        "MSC": r"\textbf{2020 Mathematics Subject Classification.} " + ", ".join(article["msc"]) + "." if article["msc"] else "",
        "KEYWORDS": escape(", ".join(article["keywords"])),
        "ABSTRACT": render_text(article["abstract"]["text"], ledger, labels, aliases), "BODY": "\n\n".join(body)}
    for key, value in substitutions.items():
        marker = "@@HYPROVER_" + key + "@@"
        if source.count(marker) != 1:
            raise ArticleError("article_template_slot_mismatch")
        source = source.replace(marker, value)
    if "@@HYPROVER_" in source:
        raise ArticleError("unfilled_article_template")
    refs, bib = bibliography(ledger, used, aliases)
    return {"paper.tex": source, "references.tex": refs, "references.bib": bib}


def write(root: Path, outputs: dict[str, str]) -> None:
    for name, content in outputs.items():
        if name not in ("paper.tex", "references.tex", "references.bib"):
            raise ArticleError("unexpected_article_output")
        atomic_write_text(safe_path(root, name), content)


def pdf_sources(root: Path) -> dict:
    return {name: text_hash(read_text(safe_path(root, name))) for name in ("paper.tex", "references.tex")}


def current_pdf(root: Path) -> dict | None:
    try:
        report = read_json(safe_path(root, "compilation.json"))
        path = safe_path(root, "paper.pdf")
        if (report.get("format") != "hyprover-article-pdf-v1" or report.get("compiled") is not True
                or report.get("source_hashes") != pdf_sources(root) or path.stat().st_size > 33554432):
            return None
        raw = path.read_bytes()
        if (not raw.startswith(b"%PDF-") or not raw.rstrip().endswith(b"%%EOF")
                or hashlib.sha256(raw).hexdigest() != report.get("pdf_sha256")):
            return None
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None


def compile_pdf(root: Path, timeout: int = 270, *, deadline: float | None = None, cancelled=None) -> dict:
    """Opt-in only. No unsafe fallback when kernel isolation is unavailable."""
    budget = Budget(timeout, "isolated_tex_compilation_unavailable_or_timed_out",
                    deadline=deadline, cancelled=cancelled)
    tools = {name: shutil.which(name) for name in ("bwrap", "pdflatex", "prlimit")}
    if not all(tools.values()):
        return {"compiled": False, "error": "safe_tex_environment_unavailable"}
    source_hashes = pdf_sources(root)
    with tempfile.TemporaryDirectory(prefix=".tex-", dir=root) as temporary:
        work = Path(temporary)
        for name in ("paper.tex", "references.tex"):
            shutil.copyfile(safe_path(root, name), work / name)
        cmd = [tools["prlimit"], "--cpu=180", "--as=1073741824", "--fsize=33554432", "--nproc=64",
               tools["bwrap"], "--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
        for location in ("/usr", "/lib", "/lib64", "/etc/texmf", "/etc/texlive", "/var/lib/texmf"):
            if Path(location).exists():
                cmd += ["--ro-bind", location, location]
        cmd += ["--tmpfs", "/tmp", "--dir", "/proc", "--dev", "/dev", "--bind", str(work), "/work",
                "--chdir", "/work", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "openin_any", "p", "--setenv", "openout_any", "p", tools["pdflatex"],
                "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "paper.tex"]
        try:
            for attempt in range(3):
                with safe_path(root, "compile.log").open("wb" if attempt == 0 else "ab") as diagnostics:
                    status = run_command(cmd, budget, stdout=diagnostics, stderr=subprocess.STDOUT)
                if status:
                    return {"compiled": False, "error": "isolated_tex_compilation_failed", "returncode": status,
                            "log_path": str(root / "compile.log")}
            output = safe_path(work, "paper.pdf")
            if not output.is_file() or output.stat().st_size > 33554432:
                return {"compiled": False, "error": "invalid_tex_output"}
            log = (work / "paper.log").read_text(errors="replace")
            if any(s in log for s in ("undefined references", "undefined citations", "LaTeX Error")):
                return {"compiled": False, "error": "unresolved_tex_references"}
            raw = output.read_bytes()
            pages = re.search(r"Output written on paper\.pdf \((\d+) pages?", log)
            if not raw.startswith(b"%PDF-") or not raw.rstrip().endswith(b"%%EOF") or not pages:
                return {"compiled": False, "error": "invalid_tex_output"}
            if pdf_sources(root) != source_hashes:
                return {"compiled": False, "error": "tex_sources_changed_during_compilation"}
            budget.remaining()
            report = {"format": "hyprover-article-pdf-v1", "compiled": True, "path": str(root / "paper.pdf"),
                      "pages": int(pages.group(1)), "bytes": len(raw), "pdf_sha256": hashlib.sha256(raw).hexdigest(),
                      "source_hashes": source_hashes, "engine": "pdflatex", "isolated": True,
                      "log_path": str(root / "compile.log"), "visual_comparison_performed": False,
                      "overfull_boxes": len(re.findall(r"Overfull \\[hv]box", log)),
                      "max_overfull_pt": max((float(v) for v in re.findall(r"Overfull \\[hv]box \(([0-9.]+)pt", log)), default=0.0)}
            output.replace(safe_path(root, "paper.pdf"))
            write_json(safe_path(root, "compilation.json"), report)
            return report
        except (OSError, OperationTimeout):
            return {"compiled": False, "error": "isolated_tex_compilation_unavailable_or_timed_out"}
