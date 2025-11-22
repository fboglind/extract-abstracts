#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_reference_abstracts.py

Given a seed PDF (e.g., an ACL paper), extract its References section, resolve each
citation to an identifier (DOI / arXiv / ACL Anthology), then fetch abstracts from:
  1) Crossref (DOI)
  2) arXiv (id)
  3) ACL Anthology JSON (id)
  4) Crossref bibliographic search (if no id)
  5) OpenAlex search (fallback or to reconstruct abstract text)

Outputs CSV + JSONL and prints a summary:
  - total references parsed
  - works resolved (got an ID)
  - abstracts retrieved

Install:
  pip install pymupdf requests tqdm python-dateutil

Usage:
  python get_reference_abstracts.py --pdf /path/to/seed.pdf \
    --out_csv refs_abstracts.csv --out_jsonl refs_abstracts.jsonl \
    --rate_limit 4
"""

import re
import os
import csv
import json
import time
import html
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import requests
from tqdm import tqdm

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise SystemExit("PyMuPDF is required. Install with: pip install pymupdf")

# ---------------------------
# Regexes & Small Utilities
# ---------------------------

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
ARXIV_RE = re.compile(r"\barXiv[: ](?P<id>\d{4}\.\d{4,5}(v\d+)?)\b", re.IGNORECASE)
# Also catch bare arXiv IDs without the "arXiv:" prefix
ARXIV_ID_RE = re.compile(r"\b(?P<id>\d{4}\.\d{4,5}(v\d+)?)\b")
# ACL Anthology IDs: e.g., P19-1234, 2020.acl-main.123, 2023.emnlp-main.999
ACL_RE = re.compile(r"\b(\d{4}\.(acl|emnlp|naacl|eacl|coling|conll|ws|semeval)-\w+\.\d+|[A-Z]\d{2}-\d{4})\b")

def strip_tags(text: str) -> str:
    # Remove simple HTML/JATS tags, unescape entities
    text = re.sub(r"<\/?(jats:)?\w+[^>]*>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return html.unescape(text)

def reconstruct_openalex_abstract(inv_idx: Dict[str, List[int]]) -> str:
    # OpenAlex provides abstract_inverted_index: { token: [positions...] }
    if not inv_idx:
        return ""
    positions = {}
    for token, poss in inv_idx.items():
        for p in poss:
            positions[p] = token
    if not positions:
        return ""
    tokens = [positions[i] for i in range(max(positions.keys()) + 1)]
    return " ".join(tokens)

def polite_get(session: requests.Session, url: str, params: dict = None, headers: dict = None, timeout=20) -> Optional[requests.Response]:
    try:
        return session.get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException:
        return None

# ---------------------------
# PDF: find and split references
# ---------------------------

def extract_references_text(pdf_path: Path) -> str:
    """Grab text from the page that contains 'References' (case-insensitive) to the end."""
    with fitz.open(pdf_path) as doc:
        full_pages = []
        start_idx = None
        for i in range(len(doc)):
            text = doc.load_page(i).get_text("text")
            full_pages.append(text)
            if start_idx is None and re.search(r"\bReferences\b", text, re.IGNORECASE):
                start_idx = i
        if start_idx is None:
            # Try 'Bibliography' as a fallback
            for i in range(len(doc)):
                if re.search(r"\bBibliography\b", full_pages[i], re.IGNORECASE):
                    start_idx = i
                    break
        if start_idx is None:
            return ""  # could not find references section
        return "\n".join(full_pages[start_idx:])

def split_reference_entries(ref_text: str) -> List[str]:
    """
    Split references into entries. ACL styles vary: numbered, bracketed, or just paragraph breaks.
    Heuristic: split on double newlines, then clean very short lines away.
    """
    # Normalize hyphenation across line breaks
    ref_text = ref_text.replace("\r\n", "\n").replace("\r", "\n")
    ref_text = re.sub(r"(\w)-\n(\w)", r"\1\2", ref_text)
    chunks = [c.strip() for c in re.split(r"\n\s*\n", ref_text) if c.strip()]
    # Filter out the header paragraph(s)
    chunks = [c for c in chunks if not re.match(r"^\s*(References|Bibliography)\s*$", c, re.IGNORECASE)]
    # Some entries get merged; optionally split when line starts like [12] or 12. or (2020) etc.
    refined = []
    for c in chunks:
        parts = re.split(r"\n(?=\[\d+\]\s|^\d+\.\s|^\(\d{4}\)\s)", c, flags=re.MULTILINE)
        for p in parts:
            p = re.sub(r"\s+", " ", p).strip()
            if len(p) > 25:
                refined.append(p)
    # Deduplicate near-identical entries
    uniq = []
    seen = set()
    for r in refined:
        key = r.lower()
        if key not in seen:
            uniq.append(r)
            seen.add(key)
    return uniq

# ---------------------------
# Identifier extraction & resolvers
# ---------------------------

def extract_ids_from_entry(entry: str) -> Dict[str, str]:
    ids = {}
    doi_m = DOI_RE.search(entry)
    if doi_m:
        ids["doi"] = doi_m.group(0)
    ax = ARXIV_RE.search(entry)
    if ax:
        ids["arxiv"] = ax.group("id")
    else:
        # bare arXiv id (avoid catching years/pages by requiring dot pattern)
        ax2 = ARXIV_ID_RE.search(entry)
        if ax2 and ("arxiv" not in ids):
            ids["arxiv"] = ax2.group("id")
    acl = ACL_RE.search(entry)
    if acl:
        ids["acl"] = acl.group(1)
    return ids

def crossref_by_doi(session: requests.Session, doi: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    url = f"https://api.crossref.org/works/{doi}"
    r = polite_get(session, url, headers={"User-Agent": "RefAbsFetcher/1.0 (mailto:you@example.com)"})
    if not r or r.status_code != 200:
        return None, None, None
    data = r.json().get("message", {})
    title = " ".join(data.get("title", [])).strip() or None
    abstract = strip_tags(data.get("abstract", "") or "")
    if not abstract:
        abstract = None
    return "doi:" + doi, title, abstract

def crossref_bibliographic_search(session: requests.Session, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    url = "https://api.crossref.org/works"
    r = polite_get(session, url, params={"query.bibliographic": text, "rows": 1},
                   headers={"User-Agent": "RefAbsFetcher/1.0 (mailto:you@example.com)"})
    if not r or r.status_code != 200:
        return None, None, None
    items = r.json().get("message", {}).get("items", [])
    if not items:
        return None, None, None
    it = items[0]
    doi = it.get("DOI")
    title = " ".join(it.get("title", [])).strip() or None
    abstract = strip_tags(it.get("abstract", "") or "") or None
    if doi:
        src_id = "doi:" + doi
    else:
        src_id = it.get("URL")
    return src_id, title, abstract

def arxiv_by_id(session: requests.Session, arxiv_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    url = "https://export.arxiv.org/api/query"
    r = polite_get(session, url, params={"id_list": arxiv_id})
    if not r or r.status_code != 200:
        return None, None, None
    # Very lightweight parse: summary between <summary>...</summary>
    m_title = re.search(r"<title>(.*?)</title>", r.text, re.DOTALL)
    m_sum = re.search(r"<summary>(.*?)</summary>", r.text, re.DOTALL)
    title = html.unescape((m_title.group(1) if m_title else "")).strip() or None
    abstract = html.unescape((m_sum.group(1) if m_sum else "")).strip() or None
    # The first <title> is usually "arXiv Query Results"; get the second occurrence if present
    titles = re.findall(r"<title>(.*?)</title>", r.text, re.DOTALL)
    if titles and len(titles) > 1:
        title = html.unescape(titles[1]).strip()
    return "arxiv:" + arxiv_id, title, abstract

def acl_anthology_by_id(session: requests.Session, acl_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # JSON endpoint exists for most IDs; normalize lowercase path
    url = f"https://aclanthology.org/{acl_id.lower()}.json"
    r = polite_get(session, url)
    if not r or r.status_code != 200:
        return None, None, None
    data = r.json()
    title = data.get("title")
    abstract = data.get("abstract") or None
    return "acl:" + acl_id, title, abstract

def openalex_search(session: requests.Session, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Search; prefer works with abstracts
    url = "https://api.openalex.org/works"
    r = polite_get(session, url, params={"search": text, "per_page": 1})
    if not r or r.status_code != 200:
        return None, None, None
    results = r.json().get("results", [])
    if not results:
        return None, None, None
    w = results[0]
    src_id = w.get("id")
    title = w.get("title")
    abstract = None
    if w.get("abstract_inverted_index"):
        abstract = reconstruct_openalex_abstract(w["abstract_inverted_index"])
    return src_id, title, abstract

def openalex_by_doi(session: requests.Session, doi: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    r = polite_get(session, url)
    if not r or r.status_code != 200:
        return None, None, None
    w = r.json()
    src_id = w.get("id")
    title = w.get("title")
    abstract = None
    if w.get("abstract_inverted_index"):
        abstract = reconstruct_openalex_abstract(w["abstract_inverted_index"])
    return src_id, title, abstract

# ---------------------------
# Main pipeline per entry
# ---------------------------

def resolve_and_fetch(entry: str, session: requests.Session) -> Dict[str, Any]:
    ids = extract_ids_from_entry(entry)
    title_guess = None
    src = None
    abstract = None

    # 1) DOI direct → Crossref
    if "doi" in ids:
        src, title_guess, abstract = crossref_by_doi(session, ids["doi"])
        if not abstract:
            # Try OpenAlex by DOI for abstract reconstruction
            oa_src, oa_title, oa_abs = openalex_by_doi(session, ids["doi"])
            if oa_abs:
                src = src or oa_src
                title_guess = title_guess or oa_title
                abstract = oa_abs

    # 2) arXiv
    if abstract is None and "arxiv" in ids:
        src2, title2, abs2 = arxiv_by_id(session, ids["arxiv"])
        if abs2:
            src, title_guess, abstract = src2, title2, abs2

    # 3) ACL Anthology
    if abstract is None and "acl" in ids:
        src3, title3, abs3 = acl_anthology_by_id(session, ids["acl"])
        if abs3:
            src, title_guess, abstract = src3, title3, abs3

    # 4) Crossref bibliographic search (use full entry text)
    if abstract is None and src is None:
        src4, title4, abs4 = crossref_bibliographic_search(session, entry)
        if src4:
            src, title_guess, abstract = src4, title4, abs4
            # If still no abstract, try OpenAlex search; often richer
            if abstract is None and title_guess:
                oa_src, oa_title, oa_abs = openalex_search(session, title_guess)
                if oa_abs:
                    src = oa_src or src
                    title_guess = oa_title or title_guess
                    abstract = oa_abs

    # 5) OpenAlex (last-ditch: full entry)
    if abstract is None and src is None:
        src5, title5, abs5 = openalex_search(session, entry)
        if src5:
            src, title_guess, abstract = src5, title5, abs5

    return {
        "raw_entry": entry,
        "src_id": src,
        "title": title_guess,
        "abstract": abstract
    }

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract references from a PDF and fetch their abstracts.")
    ap.add_argument("--pdf", required=True, help="Path to seed PDF")
    ap.add_argument("--out_csv", default="refs_abstracts.csv", help="CSV output path")
    ap.add_argument("--out_jsonl", default="refs_abstracts.jsonl", help="JSONL output path")
    ap.add_argument("--rate_limit", type=float, default=4.0, help="Max requests per second across services")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    ref_text = extract_references_text(pdf_path)
    if not ref_text:
        print("Could not find a References section in the PDF.")
        return

    entries = split_reference_entries(ref_text)
    if not entries:
        print("No reference entries detected after splitting.")
        return

    # Session with a simple rate limiter
    session = requests.Session()
    last_call = [0.0]
    min_interval = 1.0 / max(args.rate_limit, 0.1)

    def rate_wrap(func, *fa):
        now = time.time()
        elapsed = now - last_call[0]
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        out = func(*fa)
        last_call[0] = time.time()
        return out

    results = []
    for entry in tqdm(entries, desc="Resolving & fetching"):
        # patch the per-call functions to respect rate limit
        def _resolve():
            return resolve_and_fetch(entry, session)
        res = rate_wrap(lambda: _resolve())
        results.append(res)

    # Summaries
    total = len(results)
    resolved = sum(1 for r in results if r.get("src_id"))
    with_abs = sum(1 for r in results if (r.get("abstract") and len(r["abstract"].strip()) > 0))

    print("\n--- Summary ---")
    print(f"References parsed: {total}")
    print(f"Resolved works:    {resolved}")
    print(f"Abstracts fetched: {with_abs}")

    # Write CSV
    fieldnames = ["src_id", "title", "abstract", "raw_entry"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: (r.get(k) or "") for k in fieldnames})

    # Write JSONL
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {total} rows to:\n  {args.out_csv}\n  {args.out_jsonl}")

if __name__ == "__main__":
    main()

