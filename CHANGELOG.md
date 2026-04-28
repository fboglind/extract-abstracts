## changelog

### Recent changes

- added support for extracting abstracts from an entire folder of PDFs
- added optional dumping of normalized first-page text to `.firstpages.txt`
- added filename metadata parsing:
  - `year_from_filename`
  - `author_from_filename`
  - `title_from_filename`
- added `--parse_filename_metadata`
- added `--prefer_filename_title`
- updated CSV/JSONL output to include filename-derived metadata
- improved handling of ACL-style and non-ACL PDFs
- created a local HTML viewer for folder-based abstract results
- changed the compact viewer table to:
  - `# | author | title | year | abstract | status`
- removed filename and DOI from the compact table
- kept DOI, filename, and extraction diagnostics in the expanded details view
- added clickable titles in the viewer
- added a PDF directory picker to the viewer
- avoided hardcoded absolute paths in the viewer
- added PDF opening in a new tab from the viewer
- improved filtering in the viewer to search over filename metadata too
- improved reference-section parsing for two-column ACL papers
- improved reference splitting heuristics
- added stopping heuristics for appendix-like headings such as:
  - `Appendix`
  - `Supplementary Material`
  - ACL-style headings like `A Haystack texts`
