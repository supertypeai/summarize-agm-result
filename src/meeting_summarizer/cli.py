from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import load_idx_settings, load_settings
from .idx_client import download_pdf, fetch_announcements, filter_keyword, load_seen, save_seen
from .pdf import extract_text_from_pdf
from .summarizer import MeetingSummarizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize meeting PDFs using pdftotext + LLM API",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        help="Path to input PDF (omit when using --from-idx)",
    )
    parser.add_argument(
        "--api",
        choices=["openai", "gemini"],
        default="gemini",
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to output JSON summary (default: <pdf-stem>.summary.json)",
    )
    parser.add_argument(
        "--save-extracted-text",
        type=Path,
        help="Optional path to save extracted raw text",
    )

    idx_group = parser.add_argument_group("IDX fetch mode")
    idx_group.add_argument(
        "--from-idx",
        action="store_true",
        help="Fetch and summarize new announcements from IDX disclosure API",
    )
    idx_group.add_argument(
        "--idx-keyword",
        help="Override IDX keyword filter (default from IDX_KEYWORD env)",
    )
    idx_group.add_argument(
        "--idx-page-size",
        type=int,
        help="Number of IDX records to request per fetch (default from IDX_PAGE_SIZE env)",
    )
    idx_group.add_argument(
        "--idx-max-new",
        type=int,
        help="Process at most N new announcements in this run",
    )
    idx_group.add_argument(
        "--idx-download-dir",
        type=Path,
        default=Path("input"),
        help="Directory for downloaded announcement PDFs (default: input)",
    )
    idx_group.add_argument(
        "--idx-seen-file",
        type=Path,
        help="Path to JSON file tracking already processed announcement IDs",
    )

    return parser


def _summarize_pdf(
    pdf_path: Path,
    summarizer: MeetingSummarizer,
    output_path: Path | None,
    extracted_text_path: Path | None,
) -> Path:
    transcript = extract_text_from_pdf(pdf_path)

    if extracted_text_path:
        extracted_text_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_text_path.write_text(transcript, encoding="utf-8")

    summary = summarizer.summarize(transcript)

    final_output = output_path or pdf_path.with_suffix("").with_name(f"{pdf_path.stem}.summary.json")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_output.write_text(summary, encoding="utf-8")

    print(f"Summary written to: {final_output}")
    return final_output


def _summary_output_path(pdf_path: Path, output_dir: Path | None) -> Path | None:
    if output_dir is None:
        return None
    return output_dir / f"{pdf_path.stem}.summary.json"


def _extracted_output_path(pdf_path: Path, extracted_dir: Path | None) -> Path | None:
    if extracted_dir is None:
        return None
    return extracted_dir / f"{pdf_path.stem}.extracted.txt"


def _run_single_pdf(args: argparse.Namespace) -> int:
    if args.pdf is None:
        raise RuntimeError("Path to input PDF is required unless --from-idx is used.")

    settings = load_settings(args.api)
    summarizer = MeetingSummarizer(settings)
    _summarize_pdf(
        pdf_path=args.pdf,
        summarizer=summarizer,
        output_path=args.output,
        extracted_text_path=args.save_extracted_text,
    )
    return 0


def _run_from_idx(args: argparse.Namespace) -> int:
    if args.idx_page_size is not None and args.idx_page_size < 1:
        raise RuntimeError("--idx-page-size must be >= 1")
    if args.idx_max_new is not None and args.idx_max_new < 1:
        raise RuntimeError("--idx-max-new must be >= 1")

    idx_settings = load_idx_settings()

    keyword = (args.idx_keyword or idx_settings.keyword).strip()
    if not keyword:
        raise RuntimeError("IDX keyword cannot be empty.")

    page_size = args.idx_page_size or idx_settings.page_size
    seen_file = args.idx_seen_file or Path(idx_settings.seen_file)

    if args.output and args.output.suffix:
        raise RuntimeError("When --from-idx is used, --output must be a directory path.")
    if args.save_extracted_text and args.save_extracted_text.suffix:
        raise RuntimeError(
            "When --from-idx is used, --save-extracted-text must be a directory path."
        )

    print(
        "Fetching IDX announcements from "
        f"{idx_settings.page_url} with keyword '{keyword}'..."
    )
    announcements = fetch_announcements(
        idx_settings,
        keyword=keyword,
        page_size=page_size,
    )
    print(f"Fetched {len(announcements)} announcement(s) from IDX API")

    matches = filter_keyword(announcements, keyword)
    print(f"Found {len(matches)} announcement(s) matching '{keyword}'")

    matches_with_links = [announcement for announcement in matches if announcement.get("link")]
    skipped_without_link = len(matches) - len(matches_with_links)
    if skipped_without_link > 0:
        print(f"Skipped {skipped_without_link} matching announcement(s) without attachment links")

    seen_ids = load_seen(seen_file)
    new_matches = [a for a in matches_with_links if a.get("id") not in seen_ids]
    if args.idx_max_new is not None:
        new_matches = new_matches[: args.idx_max_new]

    print(f"{len(new_matches)} new announcement(s) to process")
    if not new_matches:
        print("No new announcements to summarize")
        return 0

    settings = load_settings(args.api)
    summarizer = MeetingSummarizer(settings)

    success = 0
    failed = 0
    for announcement in new_matches:
        title = announcement.get("title", "") or "Untitled"
        company = announcement.get("company", "") or "N/A"
        label = f"{company} - {title}"

        try:
            pdf_path = download_pdf(announcement, args.idx_download_dir, idx_settings)
            _summarize_pdf(
                pdf_path=pdf_path,
                summarizer=summarizer,
                output_path=_summary_output_path(pdf_path, args.output),
                extracted_text_path=_extracted_output_path(pdf_path, args.save_extracted_text),
            )
            seen_ids.add(str(announcement.get("id", "")))
            success += 1
            print(f"Processed: {label}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"Failed: {label} ({exc})", file=sys.stderr)

    save_seen(seen_file, seen_ids)
    print(f"Seen IDs saved to: {seen_file}")
    print(f"IDX run complete: {success} succeeded, {failed} failed")

    if success == 0 and failed > 0:
        return 1
    return 0 if failed == 0 else 1


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()

    try:
        if args.from_idx:
            return _run_from_idx(args)
        return _run_single_pdf(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
