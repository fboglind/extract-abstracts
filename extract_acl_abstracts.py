#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-extract abstracts from ACL-style PDFs (and many similar scholarly PDFs).

- Looks for "Abstract" on page 1–2 (configurable with --max_pages).
- Supports "Abstract" as a header on its own line and inline forms ("Abstract. We ...").
- Stops at common section starts like "1 Introduction", "Keywords", "CCS Concepts", etc.
- Captures DOI if present and makes a rough title guess.
- Writes CSV and JSONL outputs; can optionally dump first-pages text for debugging.

Usage:
    pip install pymupdf tqdm
    python extract_acl_abstracts.py \
        --input_dir /path/to/pdfs \
        --out_csv abstracts.csv \
        --out_jsonl abstracts.jsonl \
        --dump_txt \
        --max_pages 2
"""

import re
import os
import sys
import json
import csv
import argparse
from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


SECTION_STOP_PATTERNS = [
    r"^\s*\d+(\.\d+)*\s+introduction\s*$",   # "1 Introduction" or "1.1 Introduction"
    r"^\s*introduction\s*$",                 # sometimes unnumbered
    r"^\s*keywords?\s*:?\s*$",
    r"^\s*index\s+terms\s*:?\s*$",
    r"^\s*ccs\s+concepts\s*:?\s*$",
    r"^\s*categories\s+and\s+subject\s+descriptors\s*:?\s*$",
    r"^\s*related\s+work\s*$",
    r"^\s*background\s*$",
    r"^\s*methods?\s*$",
]
SECTION_STOP_REGEX = re.compile("|".join(SECTION_STOP_PATTERNS), re.IGNORECASE)

# A generic numbered heading like "2 Method", "3.1 Data"
NUMBERED_HEADING_REGEX = re.compile(r"^\s*\d+(\.\d+)*\s+[A-Z].{0,100}$")

# Abstract header line patterns
ABSTRACT_HEADER_REGEX = re.compile(r"^\s*abstract\s*[:.\-]?\s*$", re.IGNORECASE)
ABSTRACT_INLINE_REGEX  = re.compile(r"\babstract\s*[:.\-]?\s+", re.IGNORECASE)

# DOI pattern (useful to capture for later enrichment via Crossref, etc.)
DOI_REGEX = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")

def normalize_whitespace(text: str) -> str:
    # Keep newlines but collapse runs of spaces; convert Windows newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Fix common hyphenation across line breaks: e.g., "infor-\nmation" -> "information"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse spaces on each line
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def extract_text_first_pages(pdf_path: Path, max_pages: int = 2) -> str:
    if not fitz:
        raise RuntimeError("PyMuPDF (pymupdf) is required. Install via `pip install pymupdf`.")
    with fitz.open(pdf_path) as doc:
        text_parts = []
        for i in range(min(max_pages, len(doc))):
            page = doc.load_page(i)
            # "text" preserves a sensible reading order in most scholarly PDFs
            page_text = page.get_text("text")
            text_parts.append(page_text)
    return normalize_whitespace("\n".join(text_parts))

def find_abstract(text: str) -> tuple[str, str]:
    """
    Attempt to locate the abstract within the provided text.
    Returns (abstract_text, reason) where reason explains strategy used.
    Empty abstract_text means not found.
    """
    lines = text.split("\n")

    # Strategy A: Abstract header on its own line
    for idx, line in enumerate(lines):
        if ABSTRACT_HEADER_REGEX.match(line):
            collected = []
            blanks_in_a_row = 0
            for j in range(idx + 1, len(lines)):
                current = lines[j]
                if current.strip() == "":
                    blanks_in_a_row += 1
                    # allow one blank line inside the abstract but break on larger gaps
                    if blanks_in_a_row >= 2 and len(" ".join(collected).split()) > 25:
                        break
                    collected.append(current)
                    continue
                blanks_in_a_row = 0
                if SECTION_STOP_REGEX.match(current.strip()) or NUMBERED_HEADING_REGEX.match(current.strip()):
                    break
                collected.append(current)
            abstract = "\n".join(collected).strip()
            abstract = re.sub(r"\n{3,}", "\n\n", abstract).strip()
            return abstract, "header_block"

    # Strategy B: Inline form like "Abstract. We propose ..."
    m = ABSTRACT_INLINE_REGEX.search(text)
    if m:
        start = m.end()
        after = text[start:]
        # Find first stop marker
        stops = []
        # Stop at section stop patterns
        found = SECTION_STOP_REGEX.search(after)
        if found:
            stops.append(found.start())
        # Stop at numbered heading
        found2 = NUMBERED_HEADING_REGEX.search(after)
        if found2:
            stops.append(found2.start())
        end = min(stops) if stops else len(after)
        abstract = after[:end].strip()
        # If it includes "Introduction" early due to noise, trim there
        intro_m = re.search(r"\b1\W*Introduction\b|\bIntroduction\b", abstract, re.IGNORECASE)
        if intro_m and intro_m.start() > 100:
            abstract = abstract[:intro_m.start()].strip()
        return abstract, "inline_after_keyword"

    # Strategy C: Heuristic — grab a paragraph near the top before "Introduction"
    intro_m = re.search(r"\b1\W*Introduction\b", text, re.IGNORECASE)
    candidate_region = text[:intro_m.start()] if intro_m else text[:2000]
    paragraphs = [p.strip() for p in candidate_region.split("\n\n") if p.strip()]
    if paragraphs:
        best = max(paragraphs, key=len)
        if not ABSTRACT_HEADER_REGEX.match(best) and len(best.split()) > 25:
            return best.strip(), "heuristic_longest_pre_intro"

    return "", "not_found"

def guess_title(text: str) -> str:
    # Very rough: ACL PDFs often have title in the first 5–15 lines before authors and abstract.
    top = "\n".join(text.split("\n")[:15]).strip()
    # Remove "Proceedings of ..." boilerplate if present
    top = re.sub(r"^Proceedings of.*?\n", "", top, flags=re.IGNORECASE | re.DOTALL)
    lines = [ln.strip() for ln in top.split("\n") if ln.strip()]
    if lines:
        # Pick the longest line that doesn't look like affiliations/authors/Abstract
        candidates = [
            ln for ln in lines
            if not re.search(r"@|University|Institute|Inc\.|Laboratory|Abstract", ln, re.IGNORECASE)
        ]
        if candidates:
            return max(candidates, key=len)[:300]
    return ""

def extract_metadata(text: str) -> dict:
    meta = {}
    doi_m = DOI_REGEX.search(text)
    if doi_m:
        meta["doi"] = doi_m.group(0)
    title = guess_title(text)
    if title:
        meta["title_guess"] = title
    return meta

def process_pdf(pdf_path: Path, dump_txt: bool = False, max_pages: int = 2) -> dict:
    try:
        first_pages_text = extract_text_first_pages(pdf_path, max_pages=max_pages)
    except Exception as e:
        return {
            "filename": pdf_path.name,
            "abstract": "",
            "found_via": "error",
            "error": str(e),
            "doi": "",
            "title_guess": ""
        }
    abstract, reason = find_abstract(first_pages_text)
    meta = extract_metadata(first_pages_text)
    if dump_txt:
        txt_out = pdf_path.with_suffix(".firstpages.txt")
        try:
            txt_out.write_text(first_pages_text, encoding="utf-8")
        except Exception:
            pass
    return {
        "filename": pdf_path.name,
        "abstract": abstract,
        "found_via": reason,
        "error": "",
        "doi": meta.get("doi", ""),
        "title_guess": meta.get("title_guess", ""),
    }

def main():
    parser = argparse.ArgumentParser(description="Extract abstracts from ACL-style PDF articles in a folder.")
    parser.add_argument("--input_dir", required=True, help="Folder containing PDFs")
    parser.add_argument("--out_csv", default="abstracts.csv", help="Path to CSV output")
    parser.add_argument("--out_jsonl", default="abstracts.jsonl", help="Path to JSONL output")
    parser.add_argument("--dump_txt", action="store_true", help="Also dump first-pages text next to each PDF (.firstpages.txt)")
    parser.add_argument("--max_pages", type=int, default=2, help="How many first pages to scan (default 2)")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        print(f"Input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted([p for p in in_dir.glob("*.pdf")])
    if not pdfs:
        print(f"No PDFs found in {in_dir}", file=sys.stderr)

    rows = []
    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        row = process_pdf(pdf_path, dump_txt=args.dump_txt, max_pages=args.max_pages)
        rows.append(row)

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "doi", "title_guess", "abstract", "found_via", "error"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write JSONL
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} records to {args.out_csv} and {args.out_jsonl}")

if __name__ == "__main__":
    main()

