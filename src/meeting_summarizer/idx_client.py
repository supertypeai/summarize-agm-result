from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
import urllib3

from .config import IdxSettings

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}


def fetch_announcements(
    settings: IdxSettings,
    keyword: str | None = None,
    page_number: int = 1,
    page_size: int | None = None,
) -> list[dict[str, str]]:
    """Fetch IDX announcements using browser-like headers and optional proxy."""
    if not settings.verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    params = {
        "keywords": keyword or settings.keyword,
        "pageNumber": page_number,
        "pageSize": page_size or settings.page_size,
        "lang": settings.lang,
    }

    headers = {**_HEADERS, "Referer": settings.page_url}

    response = requests.get(
        settings.api_url,
        params=params,
        headers=headers,
        proxies=_build_proxies(settings.proxy_url),
        timeout=20,
        verify=settings.verify_ssl,
    )
    response.raise_for_status()

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("IDX API returned a non-JSON response.") from exc

    raw_items = data.get("Items") or data.get("Results") or []
    if not isinstance(raw_items, list):
        raise RuntimeError("Unexpected IDX API response shape: Items/Results is not a list.")

    announcements: list[dict[str, str]] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue

        title = str(item.get("Title") or item.get("Judul") or "").strip()
        date = str(item.get("PublishDate") or item.get("Date") or item.get("Tanggal") or "").strip()
        company = str(
            item.get("Code") or item.get("EmitenCode") or item.get("StockCode") or ""
        ).strip()

        link = _extract_attachment_link(item.get("Attachments"))

        ann_id = str(
            item.get("Id")
            or item.get("AnnouncementId")
            or item.get("NewsId")
            or link
            or title
            or f"announcement-{index}"
        ).strip()

        announcements.append(
            {
                "id": ann_id,
                "title": title,
                "date": date,
                "company": company,
                "link": link,
            }
        )

    return announcements


def filter_keyword(announcements: list[dict[str, str]], keyword: str) -> list[dict[str, str]]:
    """Return announcements whose title contains keyword (case-insensitive)."""
    normalized = keyword.lower().strip()
    if not normalized:
        return announcements

    matches: list[dict[str, str]] = []
    for announcement in announcements:
        title = announcement.get("title", "")
        if normalized in title.lower():
            matches.append(announcement)
    return matches


def load_seen(seen_file: Path) -> set[str]:
    """Load seen announcement IDs from JSON file."""
    if not seen_file.exists():
        return set()

    try:
        data = json.loads(seen_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Seen file is not valid JSON: {seen_file}") from exc

    if not isinstance(data, list):
        raise RuntimeError(f"Seen file JSON must be a list of IDs: {seen_file}")

    return {str(value) for value in data if str(value).strip()}


def save_seen(seen_file: Path, seen_ids: set[str]) -> None:
    """Persist seen announcement IDs to disk."""
    seen_file.parent.mkdir(parents=True, exist_ok=True)
    seen_file.write_text(
        json.dumps(sorted(seen_ids), indent=2),
        encoding="utf-8",
    )


def download_pdf(
    announcement: dict[str, str],
    target_dir: Path,
    settings: IdxSettings,
) -> Path:
    """Download an announcement PDF to target_dir and return local file path."""
    link = announcement.get("link", "").strip()
    if not link:
        raise RuntimeError(f"Announcement has no attachment link: {announcement.get('id', '')}")

    if not settings.verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    filename = _resolve_filename(announcement)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / filename

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    response = requests.get(
        link,
        headers={"User-Agent": _HEADERS["User-Agent"], "Referer": settings.page_url},
        proxies=_build_proxies(settings.proxy_url),
        timeout=30,
        verify=settings.verify_ssl,
        stream=True,
    )
    response.raise_for_status()

    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                handle.write(chunk)

    if destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {destination}")

    return destination


def _build_proxies(proxy_url: str | None) -> dict[str, str] | None:
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}


def _extract_attachment_link(attachments: object) -> str:
    if not isinstance(attachments, list):
        return ""

    preferred: str = ""
    fallback: str = ""

    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue

        raw_path = (
            attachment.get("FullSavePath")
            or attachment.get("SavePath")
            or attachment.get("Path")
            or ""
        )
        path = str(raw_path).strip()
        if not path:
            continue

        absolute = _absolute_idx_url(path)
        if absolute.lower().endswith(".pdf"):
            preferred = absolute
            break
        if not fallback:
            fallback = absolute

    return preferred or fallback


def _absolute_idx_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if path.startswith("/"):
        return f"https://www.idx.co.id{path}"
    return f"https://www.idx.co.id/{path}"


def _resolve_filename(announcement: dict[str, str]) -> str:
    url_path_name = Path(unquote(urlparse(announcement.get("link", "")).path)).name

    if url_path_name and url_path_name.lower().endswith(".pdf"):
        base_name = _sanitize_filename(url_path_name)
    else:
        title = _sanitize_filename(announcement.get("title", "announcement"))
        ann_id = _sanitize_filename(announcement.get("id", ""))
        stem = title or ann_id or "announcement"
        base_name = f"{stem}.pdf"

    # Ensure .pdf extension to keep downstream processing explicit.
    if not base_name.lower().endswith(".pdf"):
        base_name = f"{base_name}.pdf"

    return base_name


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1F]+", "_", value).strip(" .")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:180]
