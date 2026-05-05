from __future__ import annotations

import json
import re
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

import requests
import urllib3

from .config import IdxSettings

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}


def fetch_announcements(
    settings: IdxSettings,
    keyword: str | None = None,
    page_number: int = 1,
    page_size: int | None = None,
    summary_mode: Literal["agms", "pubex"] = "agms",
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

    response = _idx_get_with_fallback(
        url=settings.api_url,
        headers=headers,
        proxies=_build_proxies(settings.proxy_url),
        timeout=20,
        verify_ssl=settings.verify_ssl,
        params=params,
    )

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

        link = _extract_attachment_link(
            item.get("Attachments"),
            summary_mode=summary_mode,
        )

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
                "attachments": item.get("Attachments") if isinstance(item.get("Attachments"), list) else [],
            }
        )

    return announcements


def filter_keyword(announcements: list[dict[str, str]], keyword: str) -> list[dict[str, str]]:
    """Return announcements matching keyword with substring or wildcard support.

    - Without wildcard chars (`*`, `?`): case-insensitive substring match.
    - With wildcard chars: case-insensitive wildcard match where:
      * `*` matches any sequence of characters
      * `?` matches any single character
    """
    normalized = keyword.strip()
    if not normalized:
        return announcements

    matches: list[dict[str, str]] = []
    for announcement in announcements:
        title = str(announcement.get("title", ""))
        if _matches_pattern(title, normalized):
            matches.append(announcement)
    return matches


def filter_company(announcements: list[dict[str, str]], company: str) -> list[dict[str, str]]:
    """Return announcements matching company code with substring/wildcard support."""
    normalized = company.strip()
    if not normalized:
        return announcements

    matches: list[dict[str, str]] = []
    for announcement in announcements:
        value = str(announcement.get("company", ""))
        if _matches_pattern(value, normalized):
            matches.append(announcement)
    return matches


def filter_companies(
    announcements: list[dict[str, str]],
    companies: list[str],
) -> list[dict[str, str]]:
    """Return announcements matching any provided company filter."""
    normalized = [value.strip() for value in companies if value.strip()]
    if not normalized:
        return announcements

    matches: list[dict[str, str]] = []
    for announcement in announcements:
        value = str(announcement.get("company", ""))
        if any(_matches_pattern(value, pattern) for pattern in normalized):
            matches.append(announcement)
    return matches


def parse_announcement_date(raw_date: str | None) -> date | None:
    if not raw_date:
        return None

    value = str(raw_date).strip()
    if not value:
        return None

    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass

    if "T" in value:
        value = value.split("T", maxsplit=1)[0]

    try:
        return date.fromisoformat(value)
    except ValueError:
        pass

    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    return None


def filter_by_date_range(
    announcements: list[dict[str, str]],
    since: date | None = None,
    until: date | None = None,
) -> list[dict[str, str]]:
    if since is None and until is None:
        return announcements

    filtered: list[dict[str, str]] = []
    for announcement in announcements:
        publish_date = parse_announcement_date(announcement.get("date"))
        if publish_date is None:
            continue
        if since and publish_date < since:
            continue
        if until and publish_date > until:
            continue
        filtered.append(announcement)

    return filtered


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
    summary_mode: Literal["agms", "pubex"] = "agms",
) -> Path:
    """Download an announcement PDF to target_dir and return local file path."""
    if summary_mode == "pubex":
        pubex_path = _download_best_pubex_pdf(announcement, target_dir, settings)
        if pubex_path is not None:
            return pubex_path

    link = announcement.get("link", "").strip()
    if not link:
        raise RuntimeError(f"Announcement has no attachment link: {announcement.get('id', '')}")

    return _download_pdf_link(
        link=link,
        target_dir=target_dir,
        settings=settings,
        announcement=announcement,
    )


def _download_pdf_link(
    *,
    link: str,
    target_dir: Path,
    settings: IdxSettings,
    announcement: dict[str, str],
    ordinal: int | None = None,
) -> Path:
    if not settings.verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    filename = _resolve_filename_for_link(link, announcement, ordinal=ordinal)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / filename

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    response = _idx_get_with_fallback(
        url=link.strip(),
        headers={"User-Agent": _HEADERS["User-Agent"], "Referer": settings.page_url},
        proxies=_build_proxies(settings.proxy_url),
        timeout=30,
        verify_ssl=settings.verify_ssl,
        stream=True,
    )

    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                handle.write(chunk)

    if destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {destination}")

    return destination


def _download_best_pubex_pdf(
    announcement: dict[str, str],
    target_dir: Path,
    settings: IdxSettings,
) -> Path | None:
    candidates = _extract_attachment_pdf_candidates(announcement.get("attachments"))
    if not candidates:
        return None

    scored: list[tuple[int, int, int, int, Path]] = []
    for index, (link, attachment) in enumerate(candidates, start=1):
        path = _download_pdf_link(
            link=link,
            target_dir=target_dir,
            settings=settings,
            announcement=announcement,
            ordinal=index,
        )
        size_bytes = path.stat().st_size
        page_count = _read_pdf_page_count(path)
        lampiran_score = int(_looks_like_lampiran_attachment(link, attachment))
        threshold_score = int(size_bytes > (1024 * 1024) or page_count >= 5)
        scored.append((lampiran_score, threshold_score, page_count, size_bytes, path))

    # Prefer lampiran attachments, then size/page threshold, then more pages/larger file.
    scored.sort(reverse=True)
    return scored[0][4]


def _build_proxies(proxy_url: str | None) -> dict[str, str] | None:
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}


def _idx_get_with_fallback(
    *,
    url: str,
    headers: dict[str, str],
    proxies: dict[str, str] | None,
    timeout: int,
    verify_ssl: bool,
    params: dict[str, object] | None = None,
    stream: bool = False,
) -> requests.Response:
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        response = requests.get(
            url,
            headers=headers,
            proxies=proxies,
            timeout=timeout,
            verify=verify_ssl,
            params=params,
            stream=stream,
        )
    except requests.exceptions.SSLError:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        fallback = requests.get(
            url,
            headers=headers,
            proxies=proxies,
            timeout=timeout,
            verify=False,
            params=params,
            stream=stream,
        )
        fallback.raise_for_status()
        return fallback

    if response.status_code != 403:
        response.raise_for_status()
        return response

    # Retry once with verify=False (same approach used in buyback_notify).
    response.close()
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    fallback = requests.get(
        url,
        headers=headers,
        proxies=proxies,
        timeout=timeout,
        verify=False,
        params=params,
        stream=stream,
    )
    fallback.raise_for_status()
    return fallback


def _extract_attachment_link(
    attachments: object,
    summary_mode: Literal["agms", "pubex"] = "agms",
) -> str:
    if not isinstance(attachments, list):
        return ""

    preferred_agms: str = ""
    pubex_ranked: list[tuple[int, int, int, str]] = []
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
            if not preferred_agms:
                preferred_agms = absolute

            if summary_mode == "pubex":
                size_bytes = _extract_attachment_size_bytes(attachment)
                page_count = _extract_attachment_page_count(attachment)
                is_lampiran = _looks_like_lampiran_attachment(path, attachment)
                meets_size_threshold = size_bytes > (1024 * 1024)
                meets_page_threshold = page_count >= 5
                threshold_score = int(meets_size_threshold or meets_page_threshold)
                lampiran_score = int(is_lampiran)
                pubex_ranked.append(
                    (threshold_score, lampiran_score, max(size_bytes, page_count), absolute)
                )

            # Preserve existing AGMS behavior: first PDF wins.
            if summary_mode == "agms":
                break

        if not fallback:
            fallback = absolute

    if summary_mode == "pubex" and pubex_ranked:
        # Prefer attachments that pass size/page threshold, then lampiran labels, then larger files.
        pubex_ranked.sort(reverse=True)
        return pubex_ranked[0][3]

    return preferred_agms or fallback


def _extract_attachment_pdf_candidates(attachments: object) -> list[tuple[str, dict[str, object]]]:
    if not isinstance(attachments, list):
        return []

    seen: set[str] = set()
    candidates: list[tuple[str, dict[str, object]]] = []
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
        if not absolute.lower().endswith(".pdf"):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        candidates.append((absolute, attachment))
    return candidates


def _looks_like_lampiran_attachment(path: str, attachment: dict[str, object]) -> bool:
    candidates = [path]
    for key in ("FileName", "Name", "Title", "AttachmentName", "DisplayName", "Description"):
        value = attachment.get(key)
        if value:
            candidates.append(str(value))

    combined = " ".join(candidates).lower()
    return "lampiran" in combined or re.search(r"\blamp\d*\b", combined) is not None


def _extract_attachment_size_bytes(attachment: dict[str, object]) -> int:
    for key in ("FileSizeBytes", "FileSize", "SizeBytes", "Size", "DocumentSize"):
        parsed = _parse_size_bytes(attachment.get(key))
        if parsed > 0:
            return parsed
    return 0


def _extract_attachment_page_count(attachment: dict[str, object]) -> int:
    for key in ("PageCount", "Pages", "TotalPages", "JumlahHalaman", "Page"):
        parsed = _parse_positive_int(attachment.get(key))
        if parsed > 0:
            return parsed
    return 0


def _parse_positive_int(value: object) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    normalized = re.sub(r"[^\d]", "", text)
    if not normalized:
        return 0
    return int(normalized)


def _parse_size_bytes(value: object) -> int:
    if value is None:
        return 0
    text = str(value).strip().lower()
    if not text:
        return 0

    number_match = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if number_match is None:
        return 0

    number = float(number_match.group(1).replace(",", "."))
    multiplier = 1
    if "gb" in text or "gib" in text:
        multiplier = 1024 * 1024 * 1024
    elif "mb" in text or "mib" in text:
        multiplier = 1024 * 1024
    elif "kb" in text or "kib" in text:
        multiplier = 1024
    elif "byte" in text:
        multiplier = 1
    elif number >= 10_000:
        # Likely already raw bytes.
        multiplier = 1

    return int(number * multiplier)


def _read_pdf_page_count(pdf_path: Path) -> int:
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return 0

    if result.returncode != 0:
        return 0

    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout or "", flags=re.MULTILINE)
    if not match:
        return 0
    return int(match.group(1))


def _absolute_idx_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if path.startswith("/"):
        return f"https://www.idx.co.id{path}"
    return f"https://www.idx.co.id/{path}"


def _resolve_filename(announcement: dict[str, str]) -> str:
    return _resolve_filename_for_link(announcement.get("link", ""), announcement)


def _resolve_filename_for_link(
    link: str,
    announcement: dict[str, str],
    ordinal: int | None = None,
) -> str:
    url_path_name = Path(unquote(urlparse(link).path)).name

    if url_path_name and url_path_name.lower().endswith(".pdf"):
        base_name = _sanitize_filename(url_path_name)
    else:
        title = _sanitize_filename(announcement.get("title", "announcement"))
        ann_id = _sanitize_filename(announcement.get("id", ""))
        stem = title or ann_id or "announcement"
        if ordinal is not None:
            stem = f"{stem}_{ordinal}"
        base_name = f"{stem}.pdf"

    # Ensure .pdf extension to keep downstream processing explicit.
    if not base_name.lower().endswith(".pdf"):
        base_name = f"{base_name}.pdf"

    return base_name


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1F]+", "_", value).strip(" .")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:180]


def _matches_pattern(value: str, pattern: str) -> bool:
    wildcard_mode = "*" in pattern or "?" in pattern
    if wildcard_mode:
        escaped = re.escape(pattern)
        escaped = escaped.replace(r"\*", ".*").replace(r"\?", ".")
        return re.search(escaped, value, flags=re.IGNORECASE) is not None
    return pattern.lower() in value.lower()
