from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from .config import load_idx_settings, load_settings
from .idx_client import (
    download_pdf,
    download_pubex_pdf_for_role,
    fetch_announcements,
    filter_by_date_range,
    filter_companies,
    filter_keyword,
    load_seen,
    parse_announcement_date,
    save_seen,
)
from .openrouter_vision import summarize_company_update_from_pdf
from .pdf import extract_text_from_pdf
from .summarizer import MeetingSummarizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize disclosure PDFs as AGMS agendas or Public Expose Company Update + Q&A using pdftotext + LLM API",
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
        help="Summarize as Public Expose Company Update + Q&A sections",
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
            "Filter IDX announcements by company code (e.g. BBCA or BBCA.JK, supports wildcard * and ?). "
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
    idx_group.add_argument(
        "--idx-debug-dump",
        type=Path,
        help="Write fetched IDX announcements and filter-stage results to a JSON file",
    )
    idx_group.add_argument(
        "--upsert-source-only",
        action="store_true",
        help="In --from-idx mode, update only source_file/source_link for existing AGMS rows using existing .summary.json outputs",
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


def _extract_qna_section(summary_text: str) -> str:
    lines = [line.rstrip() for line in summary_text.splitlines()]
    in_qna = False
    qna_lines: list[str] = []
    for line in lines:
        if re.match(r"^(qna|q\s*&\s*a|q/a)\s*:\s*$", line.strip(), flags=re.IGNORECASE):
            in_qna = True
            continue
        if in_qna and line.strip():
            qna_lines.append(line.strip())
    if qna_lines:
        return "\n".join(qna_lines)
    fallback = [
        line.strip()
        for line in lines
        if re.match(r"^(Q\s*&\s*A|Q/A|QA)\s*#?\d+\s*:", line.strip(), flags=re.IGNORECASE)
    ]
    if fallback:
        return "\n".join(fallback)
    return "Q&A #1: Q: Not stated. A: Not stated."


def _summarize_pubex_pair(
    qna_pdf: Path,
    company_update_pdf: Path | None,
    summarizer: MeetingSummarizer,
    output_path: Path | None,
    extracted_text_path: Path | None,
    bypass_existing: bool = False,
    symbol_override: str | None = None,
) -> Path:
    final_output = output_path or qna_pdf.with_suffix("").with_name(f"{qna_pdf.stem}.summary.json")
    if not bypass_existing and final_output.exists() and final_output.stat().st_size > 0:
        print(f"Summary already exists, skipping regenerate: {final_output}")
        return final_output

    qna_transcript = extract_text_from_pdf(qna_pdf)
    if not qna_transcript.strip():
        raise RuntimeError("Extracted PubEx QnA transcript is empty.")

    qna_summary_payload = summarizer.summarize(
        qna_transcript,
        doc_type="pubex",
        source_name=qna_pdf.name,
        symbol_override=symbol_override,
    )
    import json

    qna_payload = json.loads(qna_summary_payload)
    qna_section = _extract_qna_section(str(qna_payload.get("summary", "")))

    if company_update_pdf is not None:
        page_extraction = summarize_company_update_from_pdf(company_update_pdf)
        company_update = summarizer.summarize_pubex_company_update_from_pages(page_extraction)
    else:
        company_update = "Not stated."

    combined_summary = f"Company Update:\n{company_update}\n\nQnA:\n{qna_section}"
    qna_payload["summary"] = combined_summary
    qna_payload["tags"] = summarizer.extract_tags(combined_summary, transcript=qna_transcript)

    if extracted_text_path:
        extracted_text_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_text_path.write_text(
            (
                f"[QnA Document: {qna_pdf.name}]\n{qna_transcript}\n\n"
                f"[Company Update Document: {company_update_pdf.name if company_update_pdf else 'Not stated'}]\n"
                f"{company_update}"
            ),
            encoding="utf-8",
        )

    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_output.write_text(json.dumps(qna_payload, ensure_ascii=False), encoding="utf-8")

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


def _apply_idx_filters(
    announcements: list[dict[str, str]],
    *,
    keyword: str,
    company_filters: list[str],
    since_date: date | None,
    until_date: date | None,
) -> list[dict[str, str]]:
    matches = filter_keyword(announcements, keyword)
    if company_filters:
        matches = filter_companies(matches, company_filters)
    matches_with_links = [announcement for announcement in matches if announcement.get("link")]
    return filter_by_date_range(matches_with_links, since=since_date, until=until_date)


def _pick_best_company_update_for_qna(
    qna_announcement: dict[str, str],
    company_update_announcements: list[dict[str, str]],
) -> dict[str, str] | None:
    qna_company = str(qna_announcement.get("company", "")).strip().upper()
    qna_date = parse_announcement_date(qna_announcement.get("date"))
    if qna_date is None:
        return None

    candidates = [
        ann
        for ann in company_update_announcements
        if str(ann.get("company", "")).strip().upper() == qna_company
    ]
    if not candidates:
        return None

    eligible: list[tuple[date, dict[str, str]]] = []
    for announcement in candidates:
        parsed = parse_announcement_date(announcement.get("date"))
        if parsed is None:
            continue
        if parsed > qna_date:
            continue
        if (qna_date - parsed).days > 14:
            continue
        eligible.append((parsed, announcement))

    if not eligible:
        return None
    # Materi must be on/before QnA; choose the nearest earlier material.
    eligible.sort(key=lambda item: item[0], reverse=True)
    return eligible[0][1]


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
    if args.upsert_source_only and not args.upsert:
        raise RuntimeError("--upsert-source-only requires --upsert.")
    if args.upsert_source_only and args.summary_mode != "agms":
        raise RuntimeError("--upsert-source-only is supported only with --agms mode.")

    if args.idx_page_size is not None and args.idx_page_size < 1:
        raise RuntimeError("--idx-page-size must be >= 1")
    if args.idx_max_new is not None and args.idx_max_new < 1:
        raise RuntimeError("--idx-max-new must be >= 1")

    idx_settings = load_idx_settings()
    since_date = _parse_cli_date(args.idx_since, "--idx-since") if args.idx_since else None
    until_date = _parse_cli_date(args.idx_until, "--idx-until") if args.idx_until else None
    if since_date and until_date and since_date > until_date:
        raise RuntimeError("--idx-since must be earlier than or equal to --idx-until.")

    page_size = args.idx_page_size or idx_settings.page_size
    seen_file = args.idx_seen_file or Path(idx_settings.seen_file)
    company_filters = _parse_company_filters(args.idx_company)

    if args.output and args.output.suffix:
        raise RuntimeError("When --from-idx is used, --output must be a directory path.")
    if args.save_extracted_text and args.save_extracted_text.suffix:
        raise RuntimeError(
            "When --from-idx is used, --save-extracted-text must be a directory path."
        )

    keyword = (args.idx_keyword or idx_settings.keyword).strip()
    if not keyword:
        raise RuntimeError("IDX keyword cannot be empty.")

    date_filtered_matches: list[dict[str, str]]
    pubex_company_updates: dict[str, dict[str, str]] = {}
    debug_dump: dict[str, object] = {
        "summary_mode": args.summary_mode,
        "keyword": keyword,
        "company_filters": company_filters,
        "since_date": since_date.isoformat() if since_date else None,
        "until_date": until_date.isoformat() if until_date else None,
        "page_size": page_size,
        "idx_page_url": idx_settings.page_url,
    }
    if args.summary_mode == "pubex":
        qna_keyword = (args.idx_keyword or os.getenv("IDX_PUBEX_QNA_KEYWORD") or "laporan hasil public expose").strip()
        company_update_keyword = (
            os.getenv("IDX_PUBEX_COMPANY_UPDATE_KEYWORD") or "materi public expose"
        ).strip()
        if not qna_keyword or not company_update_keyword:
            raise RuntimeError("PubEx keywords cannot be empty.")

        print(
            "Fetching PubEx QnA announcements from "
            f"{idx_settings.page_url} with keyword '{qna_keyword}'..."
        )
        qna_announcements = _fetch_idx_announcements(
            keyword=qna_keyword,
            page_size=page_size,
            idx_settings=idx_settings,
            since_date=since_date,
            summary_mode="pubex",
        )
        qna_matches = _apply_idx_filters(
            qna_announcements,
            keyword=qna_keyword,
            company_filters=company_filters,
            since_date=since_date,
            until_date=until_date,
        )
        print(f"Found {len(qna_matches)} PubEx QnA announcement(s)")
        debug_dump["pubex_qna_keyword"] = qna_keyword
        debug_dump["pubex_qna_announcements"] = qna_announcements
        debug_dump["pubex_qna_matches"] = qna_matches

        print(
            "Fetching PubEx Company Update announcements from "
            f"{idx_settings.page_url} with keyword '{company_update_keyword}'..."
        )
        company_announcements = _fetch_idx_announcements(
            keyword=company_update_keyword,
            page_size=page_size,
            idx_settings=idx_settings,
            since_date=since_date,
            summary_mode="pubex",
        )
        company_matches = _apply_idx_filters(
            company_announcements,
            keyword=company_update_keyword,
            company_filters=company_filters,
            since_date=since_date,
            until_date=until_date,
        )
        print(f"Found {len(company_matches)} PubEx Company Update announcement(s)")
        debug_dump["pubex_company_update_keyword"] = company_update_keyword
        debug_dump["pubex_company_announcements"] = company_announcements
        debug_dump["pubex_company_matches"] = company_matches

        date_filtered_matches = qna_matches
        for qna_announcement in qna_matches:
            qna_id = str(qna_announcement.get("id", ""))
            matched_company_update = _pick_best_company_update_for_qna(
                qna_announcement,
                company_matches,
            )
            if matched_company_update is not None:
                pubex_company_updates[qna_id] = matched_company_update
        debug_dump["pubex_company_update_pairs"] = {
            qna_id: str(company_announcement.get("id", ""))
            for qna_id, company_announcement in pubex_company_updates.items()
        }
    else:
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
        date_filtered_matches = _apply_idx_filters(
            announcements,
            keyword=keyword,
            company_filters=company_filters,
            since_date=since_date,
            until_date=until_date,
        )
        print(f"Found {len(date_filtered_matches)} announcement(s) matching keyword '{keyword}'")
        debug_dump["announcements"] = announcements
        debug_dump["matches"] = date_filtered_matches

    seen_ids = load_seen(seen_file)
    new_matches: list[dict[str, str]] = []
    resumed_seen = 0
    if args.upsert_source_only:
        new_matches = list(date_filtered_matches)
    else:
        for announcement in date_filtered_matches:
            if announcement.get("id") not in seen_ids:
                new_matches.append(announcement)
                continue

            if args.bypass_existing:
                new_matches.append(announcement)
                continue

            if args.summary_mode == "pubex":
                qna_pdf = download_pubex_pdf_for_role(
                    announcement,
                    args.idx_download_dir,
                    idx_settings,
                    role="qna",
                )
                final_output = _final_summary_path(qna_pdf, args.output)
            else:
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
    debug_dump["date_filtered_matches"] = date_filtered_matches
    debug_dump["seen_file"] = str(seen_file)
    debug_dump["seen_count"] = len(seen_ids)
    debug_dump["new_matches"] = new_matches
    debug_dump["new_count"] = len(new_matches)
    debug_dump["resumed_seen_count"] = resumed_seen
    if args.idx_debug_dump:
        _write_idx_debug_dump(args.idx_debug_dump, debug_dump)
    print(f"{len(new_matches)} new announcement(s) to process")
    if not new_matches:
        print("No new announcements to summarize")
        return 0

    pending_source_upserts: list[dict[str, object]] = []

    settings = None
    summarizer = None
    if not args.upsert_source_only:
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
            if args.upsert_source_only:
                pdf_path = download_pdf(
                    announcement,
                    args.idx_download_dir,
                    idx_settings,
                    summary_mode=args.summary_mode,
                )
                final_output = _final_summary_path(pdf_path, args.output)
                if not final_output.exists() or final_output.stat().st_size == 0:
                    raise RuntimeError(
                        f"Summary output missing for source-only upsert: {final_output}"
                    )

                import json

                summary_data = json.loads(final_output.read_text(encoding="utf-8"))
                pending_source_upserts.append(
                    {
                        "symbol": summary_data["symbol"],
                        "agm_date": summary_data["date"],
                        "source_link": str(announcement.get("link", "")).strip(),
                        "source_file": label,
                    }
                )
            elif args.summary_mode == "pubex":
                qna_pdf = download_pubex_pdf_for_role(
                    announcement,
                    args.idx_download_dir,
                    idx_settings,
                    role="qna",
                )
                company_update_announcement = pubex_company_updates.get(str(announcement.get("id", "")))
                company_update_pdf: Path | None = None
                if company_update_announcement is not None:
                    company_update_pdf = download_pubex_pdf_for_role(
                        company_update_announcement,
                        args.idx_download_dir,
                        idx_settings,
                        role="company_update",
                    )
                    print(
                        "PubEx source files: "
                        f"QnA={qna_pdf.name}, CompanyUpdate={company_update_pdf.name}"
                    )
                else:
                    print(f"PubEx source files: QnA={qna_pdf.name}, CompanyUpdate=Not stated")

                final_output = _summarize_pubex_pair(
                    qna_pdf=qna_pdf,
                    company_update_pdf=company_update_pdf,
                    summarizer=summarizer,
                    output_path=_summary_output_path(qna_pdf, args.output),
                    extracted_text_path=_extracted_output_path(qna_pdf, args.save_extracted_text),
                    bypass_existing=args.bypass_existing,
                    symbol_override=_company_to_symbol(company),
                )
                if company_update_announcement is not None:
                    seen_ids.add(str(company_update_announcement.get("id", "")))
            else:
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
                    symbol_override=None,
                )
            seen_ids.add(str(announcement.get("id", "")))
            success += 1
            print(f"Processed: {label}")

            if args.upsert and not args.upsert_source_only:
                import json

                summary_data = json.loads(final_output.read_text(encoding="utf-8"))
                row: dict[str, object] = {
                    "symbol": summary_data["symbol"],
                    "agm_date": summary_data["date"],
                    "summary": summary_data["summary"],
                    "source_link": str(announcement.get("link", "")).strip(),
                }
                if args.summary_mode == "agms":
                    row["tags"] = summary_data.get("tags", [])
                    row["source_file"] = label
                pending_upserts.append(row)

        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"Failed: {label} ({exc})", file=sys.stderr)

    if args.upsert_source_only and pending_source_upserts:
        from .supabase_client import upsert_agm_sources

        updated, skipped = upsert_agm_sources(pending_source_upserts)
        print(f"Supabase source-only write complete: {updated} updated, {skipped} skipped")
    elif args.upsert and pending_upserts:
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


def _write_idx_debug_dump(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"IDX debug dump written to: {path}")


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
