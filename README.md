# extract-abstracts

## extract_acl_abstracts.py

A Tool to batch-extract abstracts from ACL-style PDFs (and many similar scholarly PDFs).

- Looks for "Abstract" on page 1–2 (configurable with --max_pages).
- Supports "Abstract" as a header on its own line and inline forms ("Abstract. We ...").
- Stops at common section starts like "1 Introduction", "Keywords", "CCS Concepts", etc.
- Captures DOI if present and makes a rough title guess.
- Writes CSV and JSONL outputs; can optionally dump first-pages text for debugging.

Usage:

```
 pip install pymupdf tqdm python extract_acl_abstracts.py \
 	--input_dir /path/to/pdfs \
 	--out_csv abstracts.csv \
 	--out_jsonl abstracts.jsonl \
 	--dump_txt \
 	--max_pages 2
```



## get_reference_abstracts.py

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

```
python get_reference_abstracts.py --pdf /path/to/seed.pdf \
	--out_csv refs_abstracts.csv --out_jsonl refs_abstracts.jsonl \
	--rate_limit 4
```



## abstract_viewer.html

A viewer for outputted .jsonl file from `get_reference_abstracts.py` locally in Firefox. 

Lets you:

- Load the refs_abstracts.jsonl file via a file picker
- Show a summary (how many refs, how many abstracts found)
- Browse/search/filter the references
- Expand a row to see full abstract + original reference text

Usage:
    - `firefox abstract-viewer.html`
        - Click “Load JSONL file” and select .jsonl-file.
        - Use the Filter box to search titles/abstracts/raw references.
        - Click any row to expand/collapse full abstract + original reference.
