# extract-abstracts

Utilities for extracting abstracts from scholarly PDFs and browsing the results locally in Firefox. Was initially created for ACL-style pdfs, but now works pretty good for other formats as well

### `extract_abstracts.py`

Batch-extracts abstracts from a folder of PDFs.

- scans a folder of `.pdf` files
- extracts text from the first pages with PyMuPDF
- detects abstracts
- guesses metadata using heuristics or can be set to parse them from filename convention: `YEAR_AUTHOR_TITLE.pdf`

Parsed filename metadata:

- `year_from_filename`
- `author_from_filename`
- `title_from_filename`

Useful flags:

- `--dump_txt`
- `--max_pages 2`
- `--parse_filename_metadata`
- `--prefer_filename_title`

Example:

```bash
python extract_abstracts.py \
  --input_dir /home/fredrik/Documents/thesis-lit \
  --out_csv thesis-abstracts.csv \
  --out_jsonl thesis-abstracts.jsonl \
  --dump_txt \
  --max_pages 2 \
  --parse_filename_metadata \
  --prefer_filename_title
```

### `get_reference_abstracts.py`

Extracts the references from a seed paper and tries to retrieve abstracts for the cited works.

- detects the References section in a PDF
- splits ACL-style references with heuristics tailored to two-column papers
- stops before appendix-like sections
- tries multiple resolution/fetching strategies:
  - DOI via Crossref
  - arXiv
  - ACL Anthology JSON
  - Crossref bibliographic search
  - OpenAlex fallback
- writes CSV and JSONL
- prints summary counts such as:
  - references parsed
  - resolved works
  - abstracts fetched

Example:

```bash
python get_reference_abstracts.py \
  --pdf 2025.emnlp-main.1742.pdf \
  --out_csv needles_refs.csv \
  --out_jsonl needles_refs.jsonl
```

### `abstract_viewer_updated.html`

A local HTML viewer for JSONL output from `extract_abstracts.py`.

- opens locally in Firefox
- loads JSONL through a file picker
- loads the PDF directory through a directory picker
- shows a compact table with:
  - index
  - author
  - title
  - year
  - abstract preview
  - status
- expands rows to show full details
- supports filtering
- opens the matching PDF in a new tab when the title is clicked
- avoids hardcoded absolute paths by indexing PDFs from a selected folder

Workflow:

1. Open `abstract_viewer_updated.html` in Firefox.
2. Load the JSONL file produced by `extract_abstracts.py`.
3. Load the directory containing the original PDFs.
4. Click a title to open its PDF.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pymupdf tqdm requests python-dateutil
```

Check that PyMuPDF works:

```bash
python -c "import fitz; print('PyMuPDF OK')"
```

## Output formats

### `extract_abstracts.py` JSONL / CSV fields

- `filename`
- `year_from_filename`
- `author_from_filename`
- `title_from_filename`
- `doi`
- `title_guess`
- `abstract`
- `found_via`
- `error`

### `get_reference_abstracts.py` JSONL / CSV fields

- `raw_entry`
- `src_id`
- `title`
- `abstract`



### Why the viewer needs the PDF folder separately

The HTML viewer does not hardcode a base path. Instead, it indexes the user-selected PDF folder and matches files by filename. This makes the setup more portable across machines and folders.

## Suggested file layout

```text
project/
├── extract_abstracts.py
├── get_reference_abstracts.py
├── abstract_viewer_updated.html
├── README.md
└── venv/
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

- open `abstract_viewer_updated.html`
- load `thesis-abstracts.jsonl`
- load the PDF folder
- click a title to open the paper

### Extract abstracts of cited references from one paper

```bash
python get_reference_abstracts.py \
  --pdf name-of-paper.pdf \
  --out_csv name-of-paper_refs.csv \
  --out_jsonl name-of-paper_refs.jsonl
```

