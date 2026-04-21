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
- Includes IDX fetch mode to auto-process new announcements matching a keyword (no email step)
- Produces JSON output with this structure:
  - `symbol`: parsed from `Kode Emiten` and normalized to `.JK`
  - `date`: meeting date (`yyyy-mm-dd`) parsed from RUPS meeting date text
  - `summary`: agenda lines only (`Agenda #1: ...` to `Agenda #n: ...`) without title

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
python -m meeting_summarizer.cli path/to/meeting.pdf --api gemini
```

Windows example with spaces in filename:

```bash
python -m meeting_summarizer.cli "input/20260325_SBMA_Ringkasan Risalah__Risalah RUPS_32055063_lamp1.pdf" --api openai
```

### Run with installed script

```bash
meeting-summarizer path/to/meeting.pdf --api gemini
```

### Save extracted text for debugging

```bash
meeting-summarizer path/to/meeting.pdf --save-extracted-text output/meeting.txt
```

### Custom output file

```bash
meeting-summarizer path/to/meeting.pdf -o output/meeting-summary.json
```

### Fetch from IDX and summarize new `ringkasan risalah` PDFs

```bash
meeting-summarizer --from-idx --api gemini
```

Useful flags for IDX mode:

- `--idx-keyword`: override `IDX_KEYWORD`
- `--idx-page-size`: override `IDX_PAGE_SIZE`
- `--idx-max-new`: process only first N new announcements
- `--idx-download-dir`: where downloaded PDFs are stored (default `input`)
- `--idx-seen-file`: JSON file storing processed announcement IDs

When `--from-idx` is used:

- `--output` is treated as an output directory (one `.summary.json` per PDF)
- `--save-extracted-text` is treated as a directory (one `.extracted.txt` per PDF)

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
