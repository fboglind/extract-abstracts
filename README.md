# extract-abstracts

Utilities for extracting abstracts from scholarly PDFs and browsing results locally in Firefox. Was initially created for ACL-style pdfs, but now works pretty good for other formats as well

### `extract_abstracts.py`

Batch-extracts abstracts from a folder of PDFs.

- scans a folder of `.pdf` files
- extracts text from the first pages with PyMuPDF
- detects abstracts
- guesses metadata using heuristics or parses them from filename

Example:

```bash
python extract_abstracts.py \
  --input_dir /home/fredrik/Documents/thesis-lit \  # Extract abstracts from pdf articles in this folder
  --out_csv thesis-abstracts.csv \                  # Path to CSV output
  --out_jsonl thesis-abstracts.jsonl \              # Path to JSONL output
  --dump_txt \                                      # Dumps first-pages text next to each PDF
  --max_pages 2 \                                   # How many first pages to scan (default 2)
  --parse_filename_metadata \                       # Parse year/author/title from custom filename format: YEAR_AUTHOR_TITLE.pdf
  --prefer_filename_title                           # Use title parsed from filename instead of PDF title guess when available
```

### `get_reference_abstracts.py`

Extracts the references from a seed paper and tries to retrieve abstracts for the cited works.

Example:

```bash
python get_reference_abstracts.py \
  --pdf name-of-paper.pdf \
  --out_csv name-of-paper-refs.csv \
  --out_jsonl name-of-paper-refs.jsonl
```

### `abstract_viewer.html`

Local HTML viewer for JSONL output from `extract_abstracts.py`
- compact table/expands rows with details
- supports filtering
- opens PDF in a new tab when clicking title

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pymupdf tqdm requests python-dateutil
```

## Quick start

### Extract abstracts from a folder

```bash
python extract_abstracts.py \
  --input_dir /home/dirname/Documents/pdfs \
  --out_csv abstracts.csv \
  --out_jsonl abstracts.jsonl \
  --dump_txt \
  --max_pages 2 \
  --parse_filename_metadata \
  --prefer_filename_title
```

### Open results in Firefox

- open `abstract_viewer.html`
- load `thesis-abstracts.jsonl`
- load the PDF folder
- click a title to open the paper

### Extract abstracts of cited references from one paper

```bash
python get_reference_abstracts.py \
  --pdf name-of-paper.pdf \
  --out_csv name-of-paper-refs.csv \
  --out_jsonl name-of-paper-refs.jsonl
```

