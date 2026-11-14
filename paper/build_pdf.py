#!/usr/bin/env python3
"""Build paper/draft.pdf from paper/draft.md.

Pipeline: Pandoc -> DOCX -> LibreOffice -> PDF.  The DOCX stage lets the
paper compile on machines without a TeX installation.  The script also fixes
a few layout defaults that Pandoc's DOCX writer leaves noisy: page margins,
table rows that split across pages, and duplicate image captions.

Run from anywhere:

    python paper/build_pdf.py

Requirements:
    - pandoc (tested with 3.14)
    - LibreOffice (tested with 26.8)

If pandoc is installed in ~/.local/bin and not on PATH, the script finds it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MD = PAPER / "draft.md"
BIB = PAPER / "refs.bib"
OUT_PDF = PAPER / "draft.pdf"
BUILD = PAPER / "build"
DOCX = BUILD / "draft.docx"
LAYOUT = BUILD / "draft_layout.docx"


def find_tool(name: str, *fallbacks: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for fallback in fallbacks:
        path = Path(fallback).expanduser()
        if path.exists():
            return str(path)
    raise SystemExit(
        f"error: {name} not found. Install it or put it on PATH."
    )


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def add_keepnext_to_para(block: str) -> str:
    if "<w:keepNext" in block:
        return block
    if "</w:pPr>" in block:
        return block.replace("</w:pPr>", "<w:keepNext/></w:pPr>", 1)
    return block.replace("<w:p>", "<w:p><w:pPr><w:keepNext/></w:pPr>", 1)


def patch_table(match: re.Match[str]) -> str:
    """Keep table rows intact and keep the table with its caption.

    A row may not split across pages (``w:cantSplit``), and every row except
    the last gets ``w:keepNext`` on its paragraphs.  That forces LibreOffice
    to move the whole table to the next page instead of leaving a fragment.
    """
    table = match.group(0)
    rows = re.findall(r"<w:tr>.*?</w:tr>", table, re.S)
    patched = []
    for index, row in enumerate(rows):
        if "<w:trPr>" in row:
            row = row.replace("<w:trPr>", "<w:trPr><w:cantSplit/>", 1)
        else:
            row = row.replace("<w:tr>", "<w:tr><w:trPr><w:cantSplit/></w:trPr>", 1)
        if index < len(rows) - 1:
            row = re.sub(
                r"<w:p>.*?</w:p>",
                lambda p: add_keepnext_to_para(p.group(0)),
                row,
                flags=re.S,
            )
        patched.append(row)
    out = table
    for old, new in zip(rows, patched):
        out = out.replace(old, new, 1)
    return out


def patch_style(styles: str, style_id: str, size: str | None = None,
                keepnext: bool = False) -> str:
    pat = re.compile(
        r'(<w:style [^>]*w:styleId="%s"[^>]*>)(.*?)(</w:style>)'
        % re.escape(style_id),
        re.S,
    )
    match = pat.search(styles)
    if not match:
        return styles
    head, body, tail = match.group(1), match.group(2), match.group(3)
    if size:
        if "<w:rPr>" in body:
            if "<w:sz " in body:
                body = re.sub(
                    r"<w:sz [^/]*/>", f'<w:sz w:val="{size}"/>', body, count=1
                )
            else:
                body = body.replace(
                    "<w:rPr>",
                    f'<w:rPr><w:sz w:val="{size}"/><w:szCs w:val="{size}"/>',
                    1,
                )
        else:
            body += (
                f'<w:rPr><w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
            )
    if keepnext:
        if "</w:pPr>" in body:
            if "<w:keepNext" not in body:
                body = body.replace("</w:pPr>", "<w:keepNext/></w:pPr>", 1)
        else:
            body += "<w:pPr><w:keepNext/></w:pPr>"
    return styles[: match.start()] + head + body + tail + styles[match.end():]


def patch_docx(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(src) as zin:
        items = {name: zin.read(name) for name in zin.namelist()}

    doc = items["word/document.xml"].decode("utf-8")

    # 0.8 in margins; the default 1 in margins waste space and split tables.
    doc = re.sub(
        r"<w:pgMar[^/]*/>",
        '<w:pgMar w:top="1080" w:right="1152" w:bottom="1080" '
        'w:left="1152" w:header="720" w:footer="720" w:gutter="0"/>',
        doc,
        count=1,
    )

    # Pandoc repeats the image alt text as an ImageCaption paragraph.  The
    # paper already carries a numbered italic caption, so drop the duplicate.
    doc = re.sub(
        r'<w:p><w:pPr><w:pStyle w:val="ImageCaption" /></w:pPr>.*?</w:p>',
        "",
        doc,
        flags=re.S,
    )
    doc = re.sub(r'descr="[^"]*"', 'descr=""', doc)

    # Table captions are bold body paragraphs ("Table 3. ...").  Keep them
    # with the table.
    def keep_table_caption(match: re.Match[str]) -> str:
        block = match.group(0)
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", block, re.S))
        if re.match(r"\s*(Table|Box)\s+\d", text):
            block = add_keepnext_to_para(block)
        return block

    doc = re.sub(r"<w:p>.*?</w:p>", keep_table_caption, doc, flags=re.S)
    doc = re.sub(r"<w:tbl>.*?</w:tbl>", patch_table, doc, flags=re.S)
    items["word/document.xml"] = doc.encode("utf-8")

    styles = items["word/styles.xml"].decode("utf-8")
    styles = patch_style(styles, "Compact", size="18")
    styles = patch_style(styles, "TableCaption", size="18", keepnext=True)
    styles = patch_style(styles, "ImageCaption", size="18")
    items["word/styles.xml"] = styles.encode("utf-8")

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, blob in items.items():
            zout.writestr(name, blob)


def main() -> None:
    pandoc = find_tool("pandoc", "~/.local/bin/pandoc")
    soffice = find_tool("libreoffice", "/snap/bin/libreoffice")
    BUILD.mkdir(parents=True, exist_ok=True)

    run(
        [
            pandoc,
            str(MD),
            "--from",
            "markdown+tex_math_single_backslash+tex_math_double_backslash",
            "--bibliography",
            str(BIB),
            "--citeproc",
            "--resource-path",
            str(PAPER),
            "-o",
            str(DOCX),
        ]
    )
    patch_docx(DOCX, LAYOUT)
    run(
        [
            soffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(BUILD),
            str(LAYOUT),
        ]
    )

    produced = BUILD / "draft_layout.pdf"
    if not produced.exists():
        raise SystemExit(f"error: LibreOffice did not produce {produced}")
    shutil.move(str(produced), str(OUT_PDF))
    print(f"wrote {OUT_PDF.relative_to(ROOT)} ({OUT_PDF.stat().st_size / 1e6:.1f} MB)")

    if shutil.which("pdfinfo"):
        info = subprocess.run(
            ["pdfinfo", str(OUT_PDF)], capture_output=True, text=True, check=False
        )
        for line in info.stdout.splitlines():
            if line.startswith("Pages:"):
                print(line)


if __name__ == "__main__":
    main()
