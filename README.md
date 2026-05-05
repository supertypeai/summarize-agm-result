# Meeting PDF Summarizer

Summarize meeting PDFs by:

0. (Optional) Fetching latest IDX disclosures and downloading matching PDF attachments
1. Extracting text from PDF with `pdftotext` (from poppler-utils)
2. Sending extracted text to either OpenAI or Gemini (selected at runtime)
3. Writing a structured JSON summary

## Features

- Uses `poppler-utils` (`pdftotext`) for reliable PDF text extraction
- For scanned/image-based PDFs, prefers `ocrmypdf` OCR and falls back to `pdftoppm + tesseract`
- Supports long transcripts with automatic chunking and merge
- Supports two summary modes:
  - `--agms`: AGM/RUPS agenda summaries (`Agenda #1: ...`)
  - `--pubex`: Public Expose question-and-answer summaries (`Q&A #1: Q: ... A: ...`)
- Includes IDX fetch mode to auto-process new announcements matching a keyword (no email step)
- Produces JSON output with this structure:
  - `symbol`: parsed from `Kode Emiten` and normalized to `.JK`
  - `date`: document date (`yyyy-mm-dd`) parsed from AGMS/PubEx date text
- `tags` (AGMS mode only): taxonomy-based labels from `agms_tags_classification.txt`, e.g. `["dividend", "board and management", "corporate governance"]`
  - `summary`:
    - AGMS mode: agenda lines only (`Agenda #1: ...` to `Agenda #n: ...`)
    - PubEx mode: Q&A lines (`Q&A #1: Q: ... A: ...`)

PubEx symbol rule:

- In `--pubex` mode, `symbol` is parsed from filename pattern `YYYYMMDD_SYMBOL_...`.
- Example: `20260420_ASGR_Public Expose_32072185_lamp1.pdf` -> `ASGR.JK`
- In `--from-idx --pubex` mode, `symbol` is taken from IDX company code when available (e.g. `MDIA` -> `MDIA.JK`), then falls back to filename parsing.

## Requirements

- Python 3.10+
- Poppler tools (`pdftotext` and `pdftoppm`) available in PATH
- Tesseract OCR available in PATH (for scanned/image-based PDFs)
- OCRmyPDF in PATH (recommended for better scanned PDF OCR quality)
- OpenAI and/or Gemini API key (based on `--api`)

## Install

### 1) Install poppler-utils

Linux (Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

macOS (Homebrew):

```bash
brew install poppler
```

Windows options:

- Option A: install poppler via package manager, then ensure `pdftotext.exe` is in PATH
- Option B: download a poppler build for Windows and add its `bin` folder to PATH

Quick check:

```bash
pdftotext -v
pdftoppm -v
```

### 1b) Install Tesseract OCR

Linux (Debian/Ubuntu):

```bash
sudo apt-get install -y tesseract-ocr
```

macOS (Homebrew):

```bash
brew install tesseract
```

Windows:

- Install Tesseract OCR and add the installation folder to PATH.

Quick check:

```bash
tesseract --version
```

### 1c) Install OCRmyPDF (recommended)

Install OCRmyPDF so scanned PDFs are handled with better preprocessing.

Quick check:

```bash
ocrmypdf --version
```

### 2) Create Python environment and install deps

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Why `pip install -e .` is needed:

- This project uses a `src/` layout, so module execution like `python -m meeting_summarizer.cli ...` requires installing the package into your environment.

## Configuration

Copy `.env.example` to `.env` and fill values:

```env
# For --api openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4.1-mini

# For --api gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_BASE_URL=
GEMINI_MODEL=gemini-2.5-flash

SUMMARY_CHUNK_CHARS=12000
SUMMARY_TEMPERATURE=0.2
OCR_LANG=eng
OCR_ENGINE=auto

# IDX fetch mode
IDX_API_URL=https://www.idx.co.id/primary/NewsAnnouncement/GetAllAnnouncement
IDX_PAGE_URL=https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi
IDX_KEYWORD=ringkasan risalah
IDX_SEEN_FILE=seen_announcements.json
IDX_LANG=id
IDX_PAGE_SIZE=20
IDX_VERIFY_SSL=true
IDX_PROXY_URL=
```

Notes:

- Use `--api openai` or `--api gemini` when running the CLI.
- `OPENAI_BASE_URL` is optional and defaults to `https://api.openai.com/v1/chat/completions`.
- `GEMINI_BASE_URL` is optional and defaults to Gemini `generateContent` endpoint for the selected model.
- Increase `SUMMARY_CHUNK_CHARS` if your model supports large context windows.
- Set `OCR_LANG` for OCR language packs, e.g. `eng`, `ind`, or `eng+ind`.
- Set `OCR_ENGINE` to `auto` (default), `ocrmypdf`, or `tesseract`.

## Usage

### Run with module

```bash
python -m meeting_summarizer.cli path/to/meeting.pdf --agms --api gemini
```

Windows example with spaces in filename:

```bash
python -m meeting_summarizer.cli "input/20260325_SBMA_Ringkasan Risalah__Risalah RUPS_32055063_lamp1.pdf" --agms --api openai
```

Public Expose example (Q&A summary):

```bash
python -m meeting_summarizer.cli "input/new/20260420_ASGR_Public Expose_32072185_lamp1.pdf" --pubex --api gemini
```

### Run with installed script

```bash
meeting-summarizer path/to/meeting.pdf --agms --api gemini
```

### Save extracted text for debugging

```bash
meeting-summarizer path/to/meeting.pdf --save-extracted-text output/meeting.txt
```

### Custom output file

```bash
meeting-summarizer path/to/meeting.pdf -o output/meeting-summary.json
```

### Regenerate even if summary exists

```bash
meeting-summarizer path/to/meeting.pdf --bypass-existing
```

### Upsert summary to Supabase

```bash
meeting-summarizer path/to/meeting.pdf --agms --api gemini --upsert
```

With `--upsert` in AGMS mode, the payload includes `symbol`, `agm_date`, `summary`, and `tags`.
Update behavior is:
- First try exact match on `(symbol, agm_date)`.
- If no exact match, try fallback on same `symbol` with `agm_date` in `-7` to `+7` day window; if matched, update that row and set `agm_date` to the incoming summary date.
- If still not found, skip and write to `unmatched_agms.csv`.

With `--upsert` in PubEx mode (`--pubex`), the payload includes `symbol`, `agm_date` (from summary `date`), and `summary`.
Update behavior is:
- Match existing row(s) by `symbol` where `agm_place_desc = 'Public expose'` (no `agm_date` matching).
- Update matched row(s) with incoming `agm_date` and `summary`.
- If no matching row exists, skip and write to `unmatched_pubex.csv`.

### Choose summary mode

- `--agms` (default): agenda-focused summary for AGM/RUPS documents
- `--pubex`: question-and-answer summary for Public Expose documents

If no mode flag is provided, the CLI uses `--agms` behavior.

### Fetch from IDX and summarize new `ringkasan risalah` PDFs

```bash
meeting-summarizer --from-idx --agms --api gemini
```

To process Public Expose disclosures from IDX, combine `--pubex` with a Public Expose keyword:

```bash
meeting-summarizer --from-idx --pubex --idx-keyword "public expose" --api gemini
```

In `--from-idx --pubex` mode, attachment selection evaluates all PDF attachments and picks the best candidate using actual downloaded file stats: lampiran-like names (`lamp1`, `lampiran`) are prioritized, then `size > 1 MB` or `page >= 5`, then higher page count and larger file size.

Useful flags for IDX mode:

- `--idx-keyword`: override `IDX_KEYWORD` (supports wildcard `*` and `?`, e.g. `ringkasan risalah*MAPB*`)
- `--idx-company`: filter by company code (e.g. `BBCA`, supports wildcard `*` and `?`, can be repeated or comma-separated)
- `--idx-page-size`: override `IDX_PAGE_SIZE` (records per API page request)
- `--idx-max-new`: process only first N new announcements
- `--idx-since`: include announcements on/after publish date `YYYY-MM-DD`
- `--idx-until`: include announcements on/before publish date `YYYY-MM-DD`
- `--idx-download-dir`: where downloaded PDFs are stored (default `input`)
- `--idx-seen-file`: JSON file storing processed announcement IDs
- `--bypass-existing`: force regenerate summaries even when output `.summary.json` already exists
- `--upsert`: upsert generated summaries to Supabase (`idx_agm`) after summary generation

When `--from-idx` is used:

- `--output` is treated as an output directory (one `.summary.json` per PDF)
- `--save-extracted-text` is treated as a directory (one `.extracted.txt` per PDF)
- If a target `.summary.json` already exists and is non-empty, extraction/summarization is skipped and the file is reused.
- With `--bypass-existing`, existing `.summary.json` is regenerated instead of reused.
- Announcements in seen-file are automatically resumed when their target `.summary.json` is missing or empty.
- When `--idx-since` is set, the CLI auto-fetches additional IDX pages until it reaches announcements older than the `since` date.
- With `--upsert` in AGMS mode, successful summaries are written with update-only behavior (exact `symbol + agm_date`, then `symbol + date-window` fallback) to preserve other table columns.
- With `--upsert` in PubEx mode, successful summaries are written with update-only behavior (match by `symbol` + `agm_place_desc='Public expose'`, no `agm_date` match).

## GitHub Actions

This repo includes a workflow at `.github/workflows/ci.yml` that:

- Installs Poppler, Tesseract, and OCRmyPDF on `ubuntu-latest`
- Installs Python dependencies and package
- Runs smoke checks on every push and pull request
- Optionally runs full summarization on manual trigger (`workflow_dispatch`)

### Required repository secrets (for full summarization)

- `GEMINI_API_KEY` (required)
- `GEMINI_BASE_URL` (optional)
- `GEMINI_MODEL` (optional)

### Run full summarization in Actions

1. Go to GitHub Actions and run the `CI` workflow manually.
2. Set `pdf_path` to a PDF path that exists in the repository, for example `input/sample.pdf`.
3. If `GEMINI_API_KEY` is configured, the workflow generates `output/summary.json` and uploads it as artifact `meeting-summary`.

## Project Structure

```text
src/meeting_summarizer/
  cli.py          # CLI entrypoint
  config.py       # Environment config loader
  pdf.py          # pdftotext extraction wrapper
  summarizer.py   # Chunk + summarize + merge pipeline
```

## Limitations

- OCR quality depends on scan resolution, preprocessing, and language pack availability.
- Summary quality depends on transcript quality and chosen model.
