"""Stage 9: manuscript helpers. Cross-check every number in the draft against reports/tables, run the hostile-review
checks (significance claims, abbreviations, terminology, citations, markers) and convert Markdown -> LaTeX with pandoc.

Source-comment grammar used in reports/paper_draft.md (one data row per comment; several comments may follow each other):
    <!-- source: FILE | KEY | NAME=STATED; NAME=STATED -->
FILE   a CSV in reports/tables (or `figures/<png>` for a figure; then KEY and asserts are omitted)
KEY    `col=value & col=value` row filter (first unnamed column is called `index`; two-row headers are joined as `a/b`),
       or `*` for a long table with columns `quantity, value` (then NAME is the quantity)
STATED the number exactly as written in the text: `0.376`, `-0.263`, `1,925`, `55.7%` (= CSV value x 100),
       `abs(0.042)` (= |CSV value|), or `True`/`False` for boolean columns.
A stated number passes if it equals the CSV value rounded to the stated decimals AND appears in the text before the comment.
"""
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
DRAFT = ROOT / "reports" / "paper_draft.md"
BIB = ROOT / "reports" / "references.bib"
MULTI_HEADER = ("03_baseline_cv_summary_", "04_jensen_log_backtransform", "07_imerg_secondary_", "07_significance_matrix")
COMMENT = re.compile(r"<!--\s*source:\s*(.*?)\s*-->", re.S)
MINUS = {"−": "-", "–": "-", " ": " ", " ": " ", " ": " "}


def norm(text):
    for a, b in MINUS.items():
        text = text.replace(a, b)
    return text


# ---------- tables ----------
def load_table(name, tables=TABLES):
    path = Path(tables) / name
    if name.startswith(MULTI_HEADER):
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        df.columns = ["/".join(str(c) for c in col if not str(c).startswith("Unnamed")) for col in df.columns]
        return df.reset_index(names="index")
    df = pd.read_csv(path)
    first = df.columns[0]
    return df.rename(columns={first: "index"}) if str(first).startswith("Unnamed") or first == "" else df


def lookup(name, key, column, tables=TABLES):
    df = load_table(name, tables)
    if key.strip() == "*":
        row = df[df["quantity"].astype(str) == column]
        column = "value"
    else:
        row = df
        for cond in key.split("&"):
            col, val = (s.strip() for s in cond.split("=", 1))
            row = row[row[col].astype(str) == val]
    if len(row) != 1:
        raise KeyError(f"{name} | {key} | {column}: {len(row)} matching rows")
    return row[column].iloc[0]


def parse_stated(stated):
    """-> (number or bool, decimals, transform, literal that must appear in the text)."""
    s = norm(stated).strip()
    if s in ("True", "False"):
        return s == "True", None, "bool", None
    transform = "abs" if s.startswith("abs(") else "pct" if s.endswith("%") else None
    core = s[4:-1] if transform == "abs" else s.rstrip("%")
    literal = core.lstrip("+")
    num = float(literal.replace(",", ""))
    decimals = len(literal.split(".")[1]) if "." in literal else 0
    return num, decimals, transform, literal


def matches(csv_value, stated):
    num, dec, transform, _ = parse_stated(stated)
    if transform == "bool":
        return str(csv_value) == str(num)
    if isinstance(csv_value, str):                     # formatted cell such as "0.821 ± 0.009": use its first number
        csv_value = re.search(r"-?\d+(?:\.\d+)?", norm(csv_value)).group()
    v = float(csv_value)
    v = abs(v) if transform == "abs" else v * 100 if transform == "pct" else v
    return bool(np.isfinite(v)) and abs(v - num) <= 0.5 * 10 ** -dec + 1e-9


# ---------- parsing the draft ----------
def blocks(md):
    """Paragraph-like blocks with their first line number (1-based)."""
    out, start, buf = [], 1, []
    for i, line in enumerate(md.split("\n"), 1):
        if line.strip() == "":
            if buf:
                out.append((start, "\n".join(buf)))
            buf, start = [], i + 1
        else:
            buf.append(line)
    if buf:
        out.append((start, "\n".join(buf)))
    return out


def section_of(md):
    """Line number -> current '#'/'##' heading."""
    current, out = "", {}
    for i, line in enumerate(md.split("\n"), 1):
        if line.startswith("#"):
            current = line.lstrip("#").strip()
        out[i] = current
    return out


def parse_sources(md):
    rows = []
    sec = section_of(md)
    for start, text in blocks(md):
        for m in COMMENT.finditer(text):
            parts = [p.strip() for p in m.group(1).split("|")]
            before = COMMENT.sub(" ", text[:m.start()])
            line = start + text[:m.start()].count("\n")
            base = {"line": line, "section": sec[line], "file": parts[0], "key": parts[1] if len(parts) > 1 else ""}
            asserts = [a for a in (parts[2].split(";") if len(parts) > 2 else []) if a.strip()]
            if not asserts:
                rows.append({**base, "name": None, "stated": None, "text": before})
            for a in asserts:
                name, stated = (s.strip() for s in a.split("=", 1))
                rows.append({**base, "name": name, "stated": stated, "text": before})
    return pd.DataFrame(rows)


def check_sources(md, tables=TABLES):
    """One row per stated number: CSV value, rounding match, presence in the text. Figures only checked for existence."""
    out = []
    for r in parse_sources(md).to_dict("records"):
        row = {k: r[k] for k in ("line", "section", "file", "key", "name", "stated")}
        if pd.isna(r["name"]):
            path = ROOT / "reports" / r["file"] if r["file"].startswith("figures/") else Path(tables) / r["file"]
            out.append({**row, "csv_value": None, "value_ok": path.exists(), "in_text": True})
            continue
        try:
            v = lookup(r["file"], r["key"], r["name"], tables)
            ok = matches(v, r["stated"])
        except Exception as e:  # noqa: BLE001  (report, never hide)
            v, ok = f"ERROR {e}", False
        lit = parse_stated(r["stated"])[3]
        in_text = True if lit is None else lit in norm(r["text"])
        out.append({**row, "csv_value": v, "value_ok": ok, "in_text": in_text})
    return pd.DataFrame(out)


# ---------- hostile-review checks ----------
NUMBER = re.compile(r"(?<![\w.\-/])[-+]?\d[\d,]*(?:\.\d+)?")


def untraced_numbers(md, sections_prefix=("4", "Abstract"), allowed=()):
    """Numbers in the given sections that are not stated in a source comment of the same block.
    Skips headings, tables, figure lines, years 1950-2030, integers <= 10 and `allowed` protocol constants."""
    sec = section_of(md)
    out = []
    for start, text in blocks(md):
        if not sec[start].startswith(sections_prefix) or text.lstrip().startswith(("#", "|", "!", "<!--", "Table")):
            continue
        stated = {(parse_stated(m.split("=", 1)[1])[3] or "").lstrip("+-") for c in COMMENT.findall(text)
                  for m in (c.split("|")[2].split(";") if c.count("|") >= 2 else []) if "=" in m}
        body = norm(COMMENT.sub(" ", text))
        body = re.sub(r"\[@[^\]]*\]|\((?:Fig\.|Figure|Table|Section)[^)]*\)|(?:Fig\.|Figure|Table|Section|Eq\.) ?\S+",
                      " ", body)
        for m in NUMBER.finditer(body):
            lit = m.group().lstrip("+-").rstrip(",")
            val = float(lit.replace(",", ""))
            if lit in stated or lit in allowed or (val.is_integer() and (val <= 10 or 1950 <= val <= 2030)):
                continue
            out.append({"line": start, "section": sec[start], "number": m.group(), "context": body[
                max(0, m.start() - 50):m.end() + 30].replace("\n", " ")})
    return pd.DataFrame(out, columns=["line", "section", "number", "context"])


SIGNIF = re.compile(r"(?<!not )(?<!non-)(?<!in)significant(?:ly)?", re.I)


def significance_claims(md):
    """Blocks claiming significance: each needs a Holm-adjusted DM source (p_holm) or must state that the evidence is a
    bootstrap CI without multiplicity correction."""
    out = []
    sec = section_of(md)
    for start, text in blocks(md):
        if text.lstrip().startswith(("#", "|")) or not SIGNIF.search(COMMENT.sub(" ", text)):
            continue
        has_holm = "p_holm" in text
        hedged = bool(re.search(r"uncorrected|without multiplicity|no multiplicity|bootstrap CI", text, re.I))
        out.append({"line": start, "section": sec[start], "holm_source": has_holm, "hedged": hedged,
                    "ok": has_holm or hedged, "excerpt": COMMENT.sub(" ", text)[:120].replace("\n", " ")})
    return pd.DataFrame(out, columns=["line", "section", "holm_source", "hedged", "ok", "excerpt"])


ABBR = re.compile(r"(?<![\w-])[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*[a-z0-9]*(?![\w-])")
ANY_COMMENT = re.compile(r"<!--.*?-->", re.S)


def abbreviations(md, known=()):
    """Acronyms whose first use (outside comments, headings and tables) is not `... (ABBR)` and that are not defined
    earlier with `(ABBR` or listed as `known` (e.g. variable names explained in the variables table)."""
    body = ANY_COMMENT.sub(" ", md)
    body = re.sub(r"^---.*?^---", "", body, count=1, flags=re.S | re.M)       # YAML front matter
    body = re.sub(r"\[(?:CHECK|NEEDS CITATION)[^\]]*\]", " ", body)            # review markers
    lines = [l for l in body.split("\n") if not l.lstrip().startswith(("#", "|", "!["))]
    text = re.sub(r"\[@[^\]]*\]|`[^`]*`|\$[^$]*\$", " ", "\n".join(lines))
    first = {}
    for m in ABBR.finditer(text):
        first.setdefault(m.group(), m.start())
    out = []
    for abbr, pos in sorted(first.items(), key=lambda kv: kv[1]):
        if abbr in known:
            continue
        stem = abbr[:-1] if abbr.endswith("s") and abbr[:-1].isupper() else abbr      # plural "CIs"
        if stem in known:
            continue
        defined = re.search(r"\(" + re.escape(stem) + r"[\s;,):]", text)
        if not defined or defined.start() > pos + len(abbr) + 3:
            out.append({"abbreviation": abbr, "first_use": text[max(0, pos - 40):pos + 40].replace("\n", " "),
                        "defined_later": bool(defined)})
    return pd.DataFrame(out, columns=["abbreviation", "first_use", "defined_later"])


def terminology(md, variants):
    """{concept: [spellings]} -> counts of each spelling in the prose; a concept with > 1 spelling used is inconsistent."""
    body = COMMENT.sub(" ", md)
    rows = []
    for concept, spellings in variants.items():
        counts = {s: len(re.findall(re.escape(s), body)) for s in spellings}
        rows.append({"concept": concept, **{f"n[{s}]": n for s, n in counts.items()},
                     "consistent": sum(n > 0 for n in counts.values()) <= 1})
    return pd.DataFrame(rows)


def citation_keys(md):
    return sorted(set(re.findall(r"@([A-Za-z0-9_:-]+)", "\n".join(re.findall(r"\[[^\]]*@[^\]]*\]", md)))))


def bib_keys(bib=BIB):
    return sorted(re.findall(r"@\w+\{([^,]+),", Path(bib).read_text()))


def markers(md):
    sec = section_of(md)
    rows = []
    for i, line in enumerate(md.split("\n"), 1):
        for tag in ("[CHECK]", "[NEEDS CITATION]"):
            for _ in range(line.count(tag)):
                rows.append({"line": i, "section": sec[i], "marker": tag})
    return pd.DataFrame(rows, columns=["line", "section", "marker"])


# ---------- assembling ----------
def insert_block(md, name, content):
    """Replace the text between <!-- BEGIN name --> and <!-- END name --> (markers kept)."""
    pat = re.compile(rf"(<!-- BEGIN {re.escape(name)} -->)\n.*?(<!-- END {re.escape(name)} -->)", re.S)
    if not pat.search(md):
        raise KeyError(f"markers for {name} not found")
    return pat.sub(lambda m: m.group(1) + "\n" + content.strip("\n") + "\n" + m.group(2), md)


def markdown_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


LUA = r"""
local function src(text)
  return text:match('^<!%-%-%s*(source:.-)%s*%-%->$')
end
function RawBlock(el)
  if el.format == 'html' then
    local s = src(el.text)
    if s then return pandoc.RawBlock('latex', '% ' .. s) end
    return {}
  end
end
function RawInline(el)
  if el.format == 'html' then
    local s = src(el.text)
    if s then return pandoc.RawInline('latex', ' % ' .. s .. '\n') end
    return {}
  end
end
"""


def to_latex(md_path=DRAFT, tex_path=None, workdir=None):
    """pandoc Markdown -> standalone LaTeX: natbib citations, `% source:` comments kept, other HTML comments dropped."""
    md_path = Path(md_path)
    tex_path = Path(tex_path or md_path.with_suffix(".tex"))
    lua = Path(workdir or md_path.parent) / ".source_comments.lua"
    lua.write_text(LUA)
    try:
        subprocess.run(["pandoc", str(md_path), "-s", "--natbib", "--wrap=none", "--lua-filter", str(lua),
                        "-o", str(tex_path)], check=True, cwd=md_path.parent)
    finally:
        lua.unlink()
    tex = tex_path.read_text().replace("\\bibliography{references.bib}", "\\bibliography{references}")  # BibTeX adds .bib
    tex_path.write_text(tex)
    return tex_path
