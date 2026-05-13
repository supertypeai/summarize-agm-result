from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import requests

_PAGE_LINE_PATTERN = re.compile(r"^PAGE\s*(\d+)\s*:\s*(.+)$", flags=re.IGNORECASE)


def summarize_company_update_from_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise RuntimeError(f"PDF not found: {pdf_path}")

    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for Company Update multimodal processing.")

    endpoint = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1/chat/completions").strip()
    model = (os.getenv("OPENROUTER_VISION_MODEL") or "qwen/qwen3.6-plus").strip()
    timeout_sec = _read_int_env("OPENROUTER_TIMEOUT_SEC", 240)
    max_pages = _read_int_env("OPENROUTER_VISION_MAX_PAGES", 40)
    max_image_bytes = _read_int_env("OPENROUTER_MAX_IMAGE_BYTES", 28_000_000)
    max_pages_per_request = _read_int_env("OPENROUTER_MAX_PAGES_PER_REQUEST", 7)
    max_output_tokens = _read_int_env("OPENROUTER_MAX_OUTPUT_TOKENS", 12000)
    referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
    title = (os.getenv("OPENROUTER_X_TITLE") or "meeting-summarizer").strip()

    if timeout_sec < 1:
        raise RuntimeError("OPENROUTER_TIMEOUT_SEC must be >= 1.")
    if max_pages < 1:
        raise RuntimeError("OPENROUTER_VISION_MAX_PAGES must be >= 1.")
    if max_image_bytes < 1_000_000:
        raise RuntimeError("OPENROUTER_MAX_IMAGE_BYTES must be >= 1000000.")
    if max_pages_per_request < 1:
        raise RuntimeError("OPENROUTER_MAX_PAGES_PER_REQUEST must be >= 1.")
    if max_output_tokens < 256:
        raise RuntimeError("OPENROUTER_MAX_OUTPUT_TOKENS must be >= 256.")

    summary_text = _call_openrouter_pdf(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        pdf_path=pdf_path,
        timeout_sec=timeout_sec,
        max_pages=max_pages,
        max_image_bytes=max_image_bytes,
        max_pages_per_request=max_pages_per_request,
        max_output_tokens=max_output_tokens,
        referer=referer,
        title=title,
    )

    normalized = _normalize_company_update_pages(summary_text)
    if not normalized:
        return "Not stated."
    return normalized


def _call_openrouter_pdf(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    pdf_path: Path,
    timeout_sec: int,
    max_pages: int,
    max_image_bytes: int,
    max_pages_per_request: int,
    max_output_tokens: int,
    referer: str,
    title: str,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    rendered_images = _render_pdf_to_png_data_urls(pdf_path, max_pages=max_pages)
    image_batches = _batch_image_data_urls(
        rendered_images,
        max_total_bytes=max_image_bytes,
        max_pages_per_batch=max_pages_per_request,
    )

    extracted_sections: list[str] = []
    for batch in image_batches:
        first_page = batch[0][0]
        last_page = batch[-1][0]
        content_parts: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    "Read these Public Expose PDF page images and extract Company Update facts page-by-page. "
                    f"The first image in this request is PAGE {first_page} and the last image is PAGE {last_page}; "
                    "increment PAGE numbers in sequence across the images you receive. "
                    "Return only lines in this exact format: PAGE <n>: <facts>. "
                    "If a page has multiple facts, keep them in the same PAGE line separated by ' ; '. "
                    "Include only material Company Update facts and skip cover/title/section-divider pages with no substantive content. "
                    "preserve all numbers exactly as written; do not estimate; include chart/table/visual numeric/pure visual facts whenever legible; "
                    "do not skip image-only pages, because they may contain key diagrams/flows/cycles; "
                    "do not omit chart insights even if the same topic appears in body text. "
                    "If no pages in this request have material Company Update facts, return exactly: Not stated."
                ),
            }
        ]
        for _, image_data_url, _ in batch:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                }
            )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content_parts,
                }
            ],
            "temperature": 0,
            "max_tokens": max_output_tokens,
        }

        response = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(payload),
            timeout=timeout_sec,
        )
        if response.status_code >= 400:
            body = (response.text or "").strip()
            raise RuntimeError(f"OpenRouter multimodal request failed: {response.status_code} {body[:400]}")

        try:
            data = response.json()
        except ValueError as exc:
            raw_body = (response.text or "").strip()
            dump_path = _write_openrouter_non_json_dump(
                raw_body,
                pdf_path=pdf_path,
                first_page=first_page,
                last_page=last_page,
            )
            fallback_section = _extract_non_json_openrouter_section(raw_body)
            if fallback_section:
                extracted_sections.append(fallback_section)
                continue
            raise RuntimeError(
                "OpenRouter multimodal returned a non-JSON response. "
                f"Raw response was saved to: {dump_path}"
            ) from exc

        choices = data.get("choices", [])
        if not choices:
            continue
        finish_reason = str(choices[0].get("finish_reason", "")).strip().lower()
        if finish_reason == "length":
            raise RuntimeError(
                "OpenRouter output was truncated (finish_reason=length). "
                "Lower OPENROUTER_MAX_PAGES_PER_REQUEST or increase OPENROUTER_MAX_OUTPUT_TOKENS."
            )

        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            section = content.strip()
        elif isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            section = "\n".join(str(part) for part in parts if str(part).strip()).strip()
        else:
            section = ""

        if section:
            extracted_sections.append(section)

    return "\n".join(extracted_sections).strip()


def _normalize_company_update_pages(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""

    if text.startswith("```"):
        text = text.removeprefix("```").strip()
        if text.lower().startswith("markdown"):
            text = text[8:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    if text.lower() in {"not stated", "not stated."}:
        return ""

    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        candidate = raw_line.strip()
        if not candidate:
            continue
        if candidate.lower() in {"not stated", "not stated."}:
            continue
        match = _PAGE_LINE_PATTERN.match(candidate)
        if not match:
            continue
        page = int(match.group(1))
        facts = match.group(2).strip()
        if not facts:
            continue
        normalized = f"PAGE {page}: {facts}"
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(normalized)

    return "\n".join(lines)


def _extract_non_json_openrouter_section(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""

    if text.startswith("```"):
        text = text.removeprefix("```").strip()
        if text.lower().startswith("markdown"):
            text = text[8:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    normalized_pages = _normalize_company_update_pages(text)
    if normalized_pages:
        return normalized_pages

    lowered = text.lower()
    if lowered in {"not stated", "not stated."}:
        return "Not stated."

    # Ignore obvious HTML/error pages; only pass through textual content to downstream summarization.
    if "<html" in lowered or "<!doctype html" in lowered:
        return ""
    return text


def _write_openrouter_non_json_dump(
    raw_text: str,
    *,
    pdf_path: Path,
    first_page: int,
    last_page: int,
) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        prefix=f"openrouter_non_json_{pdf_path.stem}_{first_page}_{last_page}_",
        suffix=".txt",
    ) as handle:
        handle.write(raw_text)
        return Path(handle.name)


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc


def _render_pdf_to_png_data_urls(
    pdf_path: Path,
    *,
    max_pages: int,
) -> list[tuple[int, str, int]]:
    with tempfile.TemporaryDirectory(prefix="meeting_summarizer_openrouter_vision_") as temp_dir:
        output_prefix = Path(temp_dir) / "page"
        cmd = ["pdftoppm", "-f", "1", "-singlefile", "-png", str(pdf_path), str(output_prefix)]
        if max_pages > 1:
            cmd = [
                "pdftoppm",
                "-f",
                "1",
                "-l",
                str(max_pages),
                "-png",
                str(pdf_path),
                str(output_prefix),
            ]

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "pdftoppm not found in PATH. Install poppler-utils and ensure pdftoppm is available."
            ) from exc

        if result.returncode != 0:
            error_message = (result.stderr or "").strip() or "Unknown pdftoppm error"
            raise RuntimeError(f"Failed to render PDF pages for OpenRouter vision: {error_message}")

        images = sorted(Path(temp_dir).glob("page*.png"))
        if not images:
            raise RuntimeError("No PNG pages were rendered from PDF for OpenRouter vision.")

        data_urls: list[tuple[int, str, int]] = []
        for page_index, image_path in enumerate(images[:max_pages], start=1):
            image_size = image_path.stat().st_size
            image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            data_urls.append((page_index, f"data:image/png;base64,{image_data}", image_size))
        return data_urls


def _batch_image_data_urls(
    image_data_urls: list[tuple[int, str, int]],
    *,
    max_total_bytes: int,
    max_pages_per_batch: int,
) -> list[list[tuple[int, str, int]]]:
    batches: list[list[tuple[int, str, int]]] = []
    current_batch: list[tuple[int, str, int]] = []
    current_bytes = 0

    for page_number, data_url, image_bytes in image_data_urls:
        if image_bytes > max_total_bytes:
            raise RuntimeError(
                f"Rendered PAGE {page_number} image is {image_bytes} bytes, exceeding OPENROUTER_MAX_IMAGE_BYTES={max_total_bytes}. "
                "Increase OPENROUTER_MAX_IMAGE_BYTES."
            )

        if current_batch and (current_bytes + image_bytes > max_total_bytes or len(current_batch) >= max_pages_per_batch):
            batches.append(current_batch)
            current_batch = []
            current_bytes = 0

        current_batch.append((page_number, data_url, image_bytes))
        current_bytes += image_bytes

    if current_batch:
        batches.append(current_batch)

    return batches
