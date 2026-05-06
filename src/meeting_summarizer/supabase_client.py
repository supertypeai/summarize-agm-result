from __future__ import annotations

import json
import os
import csv
from datetime import date, datetime, timedelta, timezone
from supabase import create_client, Client

def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set to use --upsert.")
    return create_client(url, key)

def upsert_agm_summary(
    symbol: str,
    date: str,
    summary: str,
    tags: list[str] | None = None,
) -> tuple[int, int]:
    """Write one AGM summary/tags into idx_agm for an existing row."""
    return upsert_agm_summaries(
        [
            {
                "symbol": symbol,
                "agm_date": date,
                "summary": summary,
                "tags": tags or [],
            }
        ]
    )


def upsert_pubex_summary(
    symbol: str,
    date: str,
    summary: str,
) -> tuple[int, int]:
    """Write one Public Expose summary into idx_agm for an existing Public expose row."""
    return upsert_pubex_summaries(
        [
            {
                "symbol": symbol,
                "agm_date": date,
                "summary": summary,
            }
        ]
    )


def upsert_agm_summaries(rows: list[dict[str, object]]) -> tuple[int, int]:
    """Write AGM summaries/tags by update-only semantics on existing rows."""
    if not rows:
        return (0, 0)

    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        tags = row.get("tags")
        normalized_tags = [str(tag) for tag in tags] if isinstance(tags, list) else []
        normalized_rows.append(
            {
                "symbol": str(row.get("symbol", "")),
                "agm_date": str(row.get("agm_date", "")),
                "summary": str(row.get("summary", "")),
                "tags": normalized_tags,
            }
        )

    supabase = get_supabase_client()
    existing_pairs = _fetch_existing_pairs(supabase, normalized_rows)
    updatable_rows = [
        row
        for row in normalized_rows
        if (row["symbol"], row["agm_date"]) in existing_pairs
    ]
    missing_rows = [
        row
        for row in normalized_rows
        if (row["symbol"], row["agm_date"]) not in existing_pairs
    ]

    resolved_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    for row in missing_rows:
        resolved_date = _find_fallback_agm_date(
            supabase=supabase,
            symbol=str(row["symbol"]),
            agm_date=str(row["agm_date"]),
            window_days=7,
        )
        if resolved_date is None:
            unresolved_rows.append(row)
            continue

        resolved_rows.append({**row, "__match_agm_date": resolved_date})

    updatable_rows.extend(resolved_rows)
    missing_rows = unresolved_rows

    if missing_rows:
        _append_unmatched_rows(
            missing_rows,
            reason="Missing existing idx_agm row for (symbol, agm_date); skipped DB update.",
        )

    if not updatable_rows:
        return (0, len(missing_rows))

    updated_count = 0
    failed_updates: list[dict[str, object]] = []
    updated_on = _current_utc_timestamptz()

    for row in updatable_rows:
        try:
            response = (
                supabase.table("idx_agm")
                .update(
                    {
                        "agm_date": row["agm_date"],
                        "summary": row["summary"],
                        "tags": row["tags"],
                        "updated_on": updated_on,
                    }
                )
                .eq("symbol", row["symbol"])
                .eq("agm_date", row.get("__match_agm_date", row["agm_date"]))
                .execute()
            )
        except Exception as e:
            failed_updates.append({**row, "__reason": f"Database update failed: {e}"})
            continue

        if response.data:
            updated_count += 1
        else:
            failed_updates.append(
                {
                    **row,
                    "__reason": "No row updated for (symbol, agm_date); skipped DB update.",
                }
            )

    for row in failed_updates:
        _append_unmatched_rows(
            [{k: v for k, v in row.items() if k != "__reason"}],
            reason=str(row.get("__reason", "Database update failed.")),
        )

    skipped_total = len(missing_rows) + len(failed_updates)
    return (updated_count, skipped_total)


def upsert_pubex_summaries(rows: list[dict[str, object]]) -> tuple[int, int]:
    """Write Public Expose summaries by update-only semantics on existing Public expose rows."""
    if not rows:
        return (0, 0)

    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        normalized_rows.append(
            {
                "symbol": str(row.get("symbol", "")),
                "agm_date": str(row.get("agm_date", "")),
                "summary": str(row.get("summary", "")),
            }
        )

    supabase = get_supabase_client()
    existing_symbols = _fetch_existing_pubex_symbols(supabase, normalized_rows)
    updatable_rows = [row for row in normalized_rows if row["symbol"] in existing_symbols]
    missing_rows = [row for row in normalized_rows if row["symbol"] not in existing_symbols]

    if missing_rows:
        _append_unmatched_pubex_rows(
            missing_rows,
            reason=(
                "Missing existing idx_agm row for "
                "(symbol, agm_place_desc='Public expose'); skipped DB update."
            ),
        )

    if not updatable_rows:
        return (0, len(missing_rows))

    updated_count = 0
    failed_updates: list[dict[str, object]] = []
    updated_on = _current_utc_timestamptz()

    for row in updatable_rows:
        try:
            response = (
                supabase.table("idx_agm")
                .update(
                    {
                        "agm_date": row["agm_date"],
                        "summary": row["summary"],
                        "updated_on": updated_on,
                    }
                )
                .eq("symbol", row["symbol"])
                .eq("agm_place_desc", "Public expose")
                .execute()
            )
        except Exception as e:
            failed_updates.append({**row, "__reason": f"Database update failed: {e}"})
            continue

        if response.data:
            updated_count += 1
        else:
            failed_updates.append(
                {
                    **row,
                    "__reason": (
                        "No row updated for "
                        "(symbol, agm_place_desc='Public expose'); skipped DB update."
                    ),
                }
            )

    for row in failed_updates:
        _append_unmatched_pubex_rows(
            [{k: v for k, v in row.items() if k != "__reason"}],
            reason=str(row.get("__reason", "Database update failed.")),
        )

    skipped_total = len(missing_rows) + len(failed_updates)
    return (updated_count, skipped_total)


def _fetch_existing_pairs(
    supabase: Client,
    rows: list[dict[str, object]],
) -> set[tuple[str, str]]:
    symbols = sorted({str(row["symbol"]) for row in rows if str(row["symbol"]).strip()})
    dates = sorted({str(row["agm_date"]) for row in rows if str(row["agm_date"]).strip()})
    if not symbols or not dates:
        return set()

    response = (
        supabase.table("idx_agm")
        .select("symbol,agm_date")
        .in_("symbol", symbols)
        .in_("agm_date", dates)
        .execute()
    )

    existing: set[tuple[str, str]] = set()
    for item in response.data or []:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip()
        agm_date = str(item.get("agm_date", "")).strip()
        if symbol and agm_date:
            existing.add((symbol, agm_date))
    return existing


def _fetch_existing_pubex_symbols(
    supabase: Client,
    rows: list[dict[str, object]],
) -> set[str]:
    symbols = sorted({str(row["symbol"]) for row in rows if str(row["symbol"]).strip()})
    if not symbols:
        return set()

    response = (
        supabase.table("idx_agm")
        .select("symbol")
        .in_("symbol", symbols)
        .eq("agm_place_desc", "Public expose")
        .execute()
    )

    existing: set[str] = set()
    for item in response.data or []:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip()
        if symbol:
            existing.add(symbol)
    return existing


def _append_unmatched_rows(rows: list[dict[str, object]], reason: str) -> None:
    if not rows:
        return

    csv_file = "unmatched_agms.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["symbol", "agm_date", "summary", "tags", "reason"])
        for row in rows:
            writer.writerow(
                [
                    row["symbol"],
                    row["agm_date"],
                    row["summary"],
                    json.dumps(row["tags"], ensure_ascii=False),
                    reason,
                ]
            )


def _append_unmatched_pubex_rows(rows: list[dict[str, object]], reason: str) -> None:
    if not rows:
        return

    csv_file = "unmatched_pubex.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["symbol", "date", "summary", "reason"])
        for row in rows:
            writer.writerow(
                [
                    row["symbol"],
                    row["agm_date"],
                    row["summary"],
                    reason,
                ]
            )


def _find_fallback_agm_date(
    *,
    supabase: Client,
    symbol: str,
    agm_date: str,
    window_days: int,
) -> str | None:
    target_date = _parse_iso_calendar_date(agm_date)
    if target_date is None:
        return None

    window_start = (target_date - timedelta(days=window_days)).isoformat()
    window_end = (target_date + timedelta(days=window_days)).isoformat()
    response = (
        supabase.table("idx_agm")
        .select("agm_date")
        .eq("symbol", symbol)
        .gte("agm_date", window_start)
        .lte("agm_date", window_end)
        .execute()
    )

    candidates: list[date] = []
    for item in response.data or []:
        if not isinstance(item, dict):
            continue
        parsed = _parse_iso_calendar_date(str(item.get("agm_date", "")).strip())
        if parsed is not None:
            candidates.append(parsed)

    if not candidates:
        return None

    # Prefer the closest date; tie-break with earlier date for determinism.
    best = min(candidates, key=lambda candidate: (abs((candidate - target_date).days), candidate))
    return best.isoformat()


def _parse_iso_calendar_date(raw: str) -> date | None:
    value = raw.strip()
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _current_utc_timestamptz() -> str:
    # Keep an explicit UTC offset for Postgres timestamptz columns.
    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="microseconds")
