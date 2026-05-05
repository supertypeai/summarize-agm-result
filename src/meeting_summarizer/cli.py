from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from .config import load_idx_settings, load_settings
from .idx_client import (
    download_pdf,
    fetch_announcements,
    filter_by_date_range,
    filter_companies,
    filter_keyword,
    load_seen,
    parse_announcement_date,
    save_seen,
)
from .pdf import extract_text_from_pdf
from .summarizer import MeetingSummarizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize disclosure PDFs as AGMS agendas or Public Expose Q&A using pdftotext + LLM API",
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
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Upsert the generated summaries to Supabase db",
    )
    parser.add_argument(
        "--bypass-existing",
        action="store_true",
        help="Regenerate summary output even when target .summary.json already exists and is non-empty",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--agms",
        dest="summary_mode",
        action="store_const",
        const="agms",
        help="Summarize as AGMS agenda lines (default)",
    )
    mode_group.add_argument(
        "--pubex",
        dest="summary_mode",
        action="store_const",
        const="pubex",
        help="Summarize as Public Expose question-and-answer lines",
    )
    parser.set_defaults(summary_mode="agms")

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
        "--idx-company",
        action="append",
        help=(
            "Filter IDX announcements by company code (e.g. BBCA, supports wildcard * and ?). "
            "Can be repeated or comma-separated."
        ),
    )
    idx_group.add_argument(
        "--idx-page-size",
        type=int,
        help="Number of IDX records per API page request (default from IDX_PAGE_SIZE env)",
    )
    idx_group.add_argument(
        "--idx-max-new",
        type=int,
        help="Process at most N new announcements in this run",
    )
    idx_group.add_argument(
        "--idx-since",
        help="Only process announcements on/after this publish date (YYYY-MM-DD)",
    )
    idx_group.add_argument(
        "--idx-until",
        help="Only process announcements on/before this publish date (YYYY-MM-DD)",
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
    summary_mode: str,
    output_path: Path | None,
    extracted_text_path: Path | None,
    bypass_existing: bool = False,
    symbol_override: str | None = None,
) -> Path:
    final_output = output_path or pdf_path.with_suffix("").with_name(f"{pdf_path.stem}.summary.json")
    if not bypass_existing and final_output.exists() and final_output.stat().st_size > 0:
        print(f"Summary already exists, skipping regenerate: {final_output}")
        return final_output

    transcript = extract_text_from_pdf(pdf_path)

    if extracted_text_path:
        extracted_text_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_text_path.write_text(transcript, encoding="utf-8")

    summary = summarizer.summarize(
        transcript,
        doc_type=summary_mode,
        source_name=pdf_path.name,
        symbol_override=symbol_override,
    )

    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_output.write_text(summary, encoding="utf-8")

    print(f"Summary written to: {final_output}")
    return final_output


def _summary_output_path(pdf_path: Path, output_dir: Path | None) -> Path | None:
    if output_dir is None:
        return None
    return output_dir / f"{pdf_path.stem}.summary.json"


def _final_summary_path(pdf_path: Path, output_dir: Path | None) -> Path:
    return _summary_output_path(pdf_path, output_dir) or pdf_path.with_suffix("").with_name(
        f"{pdf_path.stem}.summary.json"
    )


def _extracted_output_path(pdf_path: Path, extracted_dir: Path | None) -> Path | None:
    if extracted_dir is None:
        return None
    return extracted_dir / f"{pdf_path.stem}.extracted.txt"


def _parse_cli_date(value: str, arg_name: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise RuntimeError(f"{arg_name} must be in YYYY-MM-DD format.") from exc


def _parse_company_filters(values: list[str] | None) -> list[str]:
    if not values:
        return []

    parsed: list[str] = []
    for value in values:
        for item in value.split(","):
            company = item.strip()
            if company:
                parsed.append(company)
    return parsed


def _company_to_symbol(company: str | None) -> str | None:
    if not company:
        return None
    normalized = str(company).strip().upper().removesuffix(".JK")
    if not re.fullmatch(r"[A-Z]{2,6}", normalized):
        return None
    return f"{normalized}.JK"


def _fetch_idx_announcements(
    *,
    keyword: str,
    page_size: int,
    idx_settings,
    since_date: date | None,
    summary_mode: str,
) -> list[dict[str, str]]:
    if since_date is None:
        return fetch_announcements(
            idx_settings,
            keyword=keyword,
            page_size=page_size,
            summary_mode=summary_mode,
        )

    all_announcements: list[dict[str, str]] = []
    page_number = 1
    while True:
        page_announcements = fetch_announcements(
            idx_settings,
            keyword=keyword,
            page_number=page_number,
            page_size=page_size,
            summary_mode=summary_mode,
        )
        if not page_announcements:
            break

        all_announcements.extend(page_announcements)
        print(f"Fetched page {page_number}: {len(page_announcements)} announcement(s)")

        page_dates = [
            parsed
            for parsed in (
                parse_announcement_date(announcement.get("date"))
                for announcement in page_announcements
            )
            if parsed is not None
        ]
        if page_dates and min(page_dates) < since_date:
            break

        page_number += 1

    return all_announcements


def _run_single_pdf(args: argparse.Namespace) -> int:
    if args.pdf is None:
        raise RuntimeError("Path to input PDF is required unless --from-idx is used.")

    settings = load_settings(args.api)
    summarizer = MeetingSummarizer(settings)
    final_output = _summarize_pdf(
        pdf_path=args.pdf,
        summarizer=summarizer,
        summary_mode=args.summary_mode,
        output_path=args.output,
        extracted_text_path=args.save_extracted_text,
        bypass_existing=args.bypass_existing,
    )
    
    if args.upsert:
        import json

        summary_data = json.loads(final_output.read_text(encoding="utf-8"))
        if args.summary_mode == "agms":
            from .supabase_client import upsert_agm_summary

            updated, skipped = upsert_agm_summary(
                symbol=summary_data["symbol"],
                date=summary_data["date"],
                summary=summary_data["summary"],
                tags=summary_data.get("tags", []),
            )
        else:
            from .supabase_client import upsert_pubex_summary

            updated, skipped = upsert_pubex_summary(
                symbol=summary_data["symbol"],
                date=summary_data["date"],
                summary=summary_data["summary"],
            )
        print(
            f"Supabase write complete for {summary_data['symbol']} @ {summary_data['date']}: "
            f"{updated} updated, {skipped} skipped"
        )
        
    return 0


def _run_from_idx(args: argparse.Namespace) -> int:
    if args.idx_page_size is not None and args.idx_page_size < 1:
        raise RuntimeError("--idx-page-size must be >= 1")
    if args.idx_max_new is not None and args.idx_max_new < 1:
        raise RuntimeError("--idx-max-new must be >= 1")

    idx_settings = load_idx_settings()
    since_date = _parse_cli_date(args.idx_since, "--idx-since") if args.idx_since else None
    until_date = _parse_cli_date(args.idx_until, "--idx-until") if args.idx_until else None
    if since_date and until_date and since_date > until_date:
        raise RuntimeError("--idx-since must be earlier than or equal to --idx-until.")

    keyword = (args.idx_keyword or idx_settings.keyword).strip()
    if not keyword:
        raise RuntimeError("IDX keyword cannot be empty.")

    page_size = args.idx_page_size or idx_settings.page_size
    seen_file = args.idx_seen_file or Path(idx_settings.seen_file)
    company_filters = _parse_company_filters(args.idx_company)

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
    announcements = _fetch_idx_announcements(
        keyword=keyword,
        page_size=page_size,
        idx_settings=idx_settings,
        since_date=since_date,
        summary_mode=args.summary_mode,
    )
    print(f"Fetched {len(announcements)} announcement(s) from IDX API")

    matches = filter_keyword(announcements, keyword)
    print(f"Found {len(matches)} announcement(s) matching keyword '{keyword}'")

    if company_filters:
        matches = filter_companies(matches, company_filters)
        print(
            f"Filtered to {len(matches)} announcement(s) for company filter(s): "
            + ", ".join(company_filters)
        )

    matches_with_links = [announcement for announcement in matches if announcement.get("link")]
    skipped_without_link = len(matches) - len(matches_with_links)
    if skipped_without_link > 0:
        print(f"Skipped {skipped_without_link} matching announcement(s) without attachment links")

    date_filtered_matches = filter_by_date_range(
        matches_with_links,
        since=since_date,
        until=until_date,
    )
    if since_date or until_date:
        window_label = f"{since_date or 'earliest'} to {until_date or 'latest'}"
        skipped_outside_window = len(matches_with_links) - len(date_filtered_matches)
        print(
            f"Date window {window_label}: {len(date_filtered_matches)} announcement(s) in range, "
            f"{skipped_outside_window} skipped"
        )

    seen_ids = load_seen(seen_file)
    new_matches: list[dict[str, str]] = []
    resumed_seen = 0
    for announcement in date_filtered_matches:
        if announcement.get("id") not in seen_ids:
            new_matches.append(announcement)
            continue

        if args.bypass_existing:
            new_matches.append(announcement)
            continue

        pdf_path = download_pdf(
            announcement,
            args.idx_download_dir,
            idx_settings,
            summary_mode=args.summary_mode,
        )
        final_output = _final_summary_path(pdf_path, args.output)
        if not final_output.exists() or final_output.stat().st_size == 0:
            new_matches.append(announcement)
            resumed_seen += 1

    if args.idx_max_new is not None:
        new_matches = new_matches[: args.idx_max_new]

    if resumed_seen > 0:
        print(
            f"Resuming {resumed_seen} seen announcement(s) with missing/empty summary output"
        )
    print(f"{len(new_matches)} new announcement(s) to process")
    if not new_matches:
        print("No new announcements to summarize")
        return 0

    settings = load_settings(args.api)
    summarizer = MeetingSummarizer(settings)
    pending_upserts: list[dict[str, object]] = []

    success = 0
    failed = 0
    for announcement in new_matches:
        title = announcement.get("title", "") or "Untitled"
        company = announcement.get("company", "") or "N/A"
        label = f"{company} - {title}"

        try:
            pdf_path = download_pdf(
                announcement,
                args.idx_download_dir,
                idx_settings,
                summary_mode=args.summary_mode,
            )
            final_output = _summarize_pdf(
                pdf_path=pdf_path,
                summarizer=summarizer,
                summary_mode=args.summary_mode,
                output_path=_summary_output_path(pdf_path, args.output),
                extracted_text_path=_extracted_output_path(pdf_path, args.save_extracted_text),
                bypass_existing=args.bypass_existing,
                symbol_override=_company_to_symbol(company) if args.summary_mode == "pubex" else None,
            )
            seen_ids.add(str(announcement.get("id", "")))
            success += 1
            print(f"Processed: {label}")
            
            if args.upsert:
                import json

                summary_data = json.loads(final_output.read_text(encoding="utf-8"))
                row: dict[str, object] = {
                    "symbol": summary_data["symbol"],
                    "agm_date": summary_data["date"],
                    "summary": summary_data["summary"],
                }
                if args.summary_mode == "agms":
                    row["tags"] = summary_data.get("tags", [])
                pending_upserts.append(row)
                
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"Failed: {label} ({exc})", file=sys.stderr)

    if args.upsert and pending_upserts:
        if args.summary_mode == "agms":
            from .supabase_client import upsert_agm_summaries

            updated, skipped = upsert_agm_summaries(pending_upserts)
        else:
            from .supabase_client import upsert_pubex_summaries

            updated, skipped = upsert_pubex_summaries(pending_upserts)
        print(f"Supabase batch write complete: {updated} updated, {skipped} skipped")

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
