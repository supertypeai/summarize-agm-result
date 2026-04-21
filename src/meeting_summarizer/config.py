from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


ApiProvider = Literal["openai", "gemini"]


@dataclass(frozen=True)
class Settings:
    api: ApiProvider
    api_key: str
    model: str
    base_url: str | None
    chunk_chars: int
    temperature: float


@dataclass(frozen=True)
class IdxSettings:
    api_url: str
    page_url: str
    keyword: str
    seen_file: str
    proxy_url: str | None
    lang: str
    page_size: int
    verify_ssl: bool


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number.") from exc


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean (true/false).")


def load_settings(api: ApiProvider) -> Settings:
    if api == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when --api openai is used.")
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        base_url = os.getenv("OPENAI_BASE_URL")
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required when --api gemini is used.")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        base_url = os.getenv("GEMINI_BASE_URL")

    chunk_chars = _read_int("SUMMARY_CHUNK_CHARS", 12000)
    temperature = _read_float("SUMMARY_TEMPERATURE", 0.2)

    return Settings(
        api=api,
        api_key=api_key,
        model=model,
        base_url=base_url,
        chunk_chars=chunk_chars,
        temperature=temperature,
    )


def load_idx_settings() -> IdxSettings:
    raw_proxy = (os.getenv("IDX_PROXY_URL") or os.getenv("PROXY_URL") or "").strip()

    return IdxSettings(
        api_url=os.getenv(
            "IDX_API_URL",
            "https://www.idx.co.id/primary/NewsAnnouncement/GetAllAnnouncement",
        ),
        page_url=os.getenv(
            "IDX_PAGE_URL",
            "https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi",
        ),
        keyword=os.getenv("IDX_KEYWORD", "ringkasan risalah"),
        seen_file=os.getenv("IDX_SEEN_FILE", "seen_announcements.json"),
        proxy_url=raw_proxy or None,
        lang=os.getenv("IDX_LANG", "id"),
        page_size=_read_int("IDX_PAGE_SIZE", 20),
        verify_ssl=_read_bool("IDX_VERIFY_SSL", True),
    )
