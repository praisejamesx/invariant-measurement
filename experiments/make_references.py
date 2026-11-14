"""Generate a readable author-year reference list from paper/refs.bib.

The BibTeX file is the source of truth.  This script exists so the Markdown
paper can carry a human-readable bibliography without requiring Pandoc.

Run:
    python experiments/make_references.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pybtex.database import parse_file

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "refs.bib"
OUT = ROOT / "paper" / "references.md"


def clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace(r"\&", "&")
    text = text.replace("---", "\u2014").replace("--", "\u2013")
    accents = {
        r'\"a': "\u00e4", r'\"o': "\u00f6", r'\"u': "\u00fc",
        r'\"A': "\u00c4", r'\"O': "\u00d6", r'\"U': "\u00dc",
        r"\'a": "\u00e1", r"\'e": "\u00e9", r"\'i": "\u00ed",
        r"\'o": "\u00f3", r"\'u": "\u00fa", r"\'c": "\u0107",
        r"\`a": "\u00e0", r"\`e": "\u00e8", r"\^a": "\u00e2",
        r"\^e": "\u00ea", r"\^o": "\u00f4", r"\~n": "\u00f1",
        r"\c{c}": "\u00e7", r"\c{c}": "\u00e7",
    }
    for old, new in accents.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\['\"`^~c]", "", text)
    text = re.sub(r"[{}]", "", text)
    return " ".join(text.split())


def format_authors(entry) -> str:
    names = []
    for person in entry.persons.get("author", []):
        last = clean(" ".join(list(person.prelast_names) + list(person.last_names)))
        first = clean(" ".join(person.first_names))
        initials = " ".join(
            [part[0] + "." for part in re.split(r"\s+", first) if part]
        ) if first else ""
        names.append(f"{last}, {initials}".strip(", "))
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def format_entry(entry) -> str:
    f = entry.fields
    bits = []
    authors = format_authors(entry).rstrip(".")
    title = clean(f.get("title", ""))
    year = clean(f.get("year", ""))
    venue = clean(
        f.get("journal")
        or f.get("booktitle")
        or f.get("publisher")
        or f.get("howpublished")
        or ""
    )
    volume = clean(f.get("volume", ""))
    number = clean(f.get("number", ""))
    pages = clean(f.get("pages", ""))
    doi = clean(f.get("doi", ""))

    if authors:
        bits.append(authors + ".")
    if year:
        bits.append(f"({year}).")
    if title:
        bits.append(f"*{title}*.")
    if venue:
        bits.append(venue + ("." if not venue.endswith(".") else ""))
    if volume:
        bits.append(
            volume
            + (f"({number})" if number else "")
            + (f", {pages}" if pages else "")
            + "."
        )
    elif pages:
        bits.append(pages + ".")
    if doi:
        bits.append(f"doi:{doi}.")
    return " ".join(bits)


def main() -> None:
    bib = parse_file(str(BIB))
    rows = []
    for key, entry in bib.entries.items():
        authors = format_authors(entry).lower()
        rows.append((authors or key, format_entry(entry)))
    rows.sort()
    lines = [
        "# References",
        "",
        "Generated from `paper/refs.bib`; the BibTeX file is the source of truth.",
        "",
    ]
    lines.extend("- " + text for _, text in rows)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(rows)} references)")


if __name__ == "__main__":
    main()
