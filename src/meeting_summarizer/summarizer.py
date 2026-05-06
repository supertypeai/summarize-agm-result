from __future__ import annotations

import json
import re
from datetime import datetime
from urllib import parse, request
from urllib.error import HTTPError, URLError
from dataclasses import dataclass
from typing import Literal

from .config import Settings

SYSTEM_PROMPT = """You are an expert meeting analyst.
Return clean text.
Be factual and only use information found in the transcript.
If information is missing, explicitly say it is not stated.
"""

SummaryDocType = Literal["agms", "pubex"]

AGMS_TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "buyback": ("buyback", "treasury shares", "treasury stock"),
    "dividend": ("cash dividend", "dividend", "bonus shares"),
    "rights issue": ("rights issue", "warrants"),
    "capital and equity": (
        "capital increase",
        "capital reduction",
        "capital structure",
        "capital utilization",
        "authorized capital",
        "share issuance",
        "share sale",
        "stock split",
        "stock options",
        "private placement",
        "public offering",
        "ipo proceeds",
        "secondary listing",
        "shareholder composition",
        "shareholder structure",
    ),
    "debt and credit": (
        "credit agreement",
        "loan facility",
        "debt conversion",
        "debt settlement",
        "bond proceeds",
        "corporate guarantee",
        "asset guarantee",
        "asset pledge",
    ),
    "acquisition and investment": (
        "acquisition",
        "divestment",
        "asset disposal",
        "asset sale",
        "investment",
        "equity investment",
        "material transaction",
    ),
    "board and management": (
        "board appointment",
        "commissioner appointment",
        "director appointment",
        "board dismissal",
        "commissioner dismissal",
        "director dismissal",
        "board resignation",
        "commissioner resignation",
        "director resignation",
        "remuneration",
        "authority",
        "composition",
        "duties",
        "discharge",
        "restructuring",
        "management change",
        "representation",
        "pension fund",
    ),
    "financial reporting and profit allocation": (
        "financial statements",
        "annual report",
        "net profit",
        "income allocation",
        "profit allocation",
        "profit utilization",
        "fund allocation",
        "fund usage",
        "fund utilization",
        "retained earnings",
        "reserve fund",
        "reserves",
        "proceeds revision",
        "proceeds use",
        "proceeds utilization",
        "working capital",
    ),
    "corporate governance": (
        "general meeting",
        "agm delay",
        "meeting minutes",
        "articles amendment",
        "notarial deed",
        "legal authorization",
        "administrative authorization",
        "liability discharge",
        "corporate plan",
        "recovery plan",
        "sustainable finance",
        "sharia board",
        "auditor appointment",
    ),
    "administrative and other": (
        "name change",
        "domicile change",
        "address change",
        "office relocation",
        "kbli compliance",
        "business activities",
        "business expansion",
        "business plan",
        "work plan",
        "feasibility study",
    ),
}

AGMS_EXTRA_CLASSIFICATION_SIGNALS: dict[str, tuple[str, ...]] = {
    "board and management": (
        "director appointment",
        "commissioner appointment",
        "remuneration",
        "discharge",
    ),
    "financial reporting and profit allocation": (
        "annual report",
        "financial statements",
        "net profit",
        "reserves",
    ),
    "corporate governance": ("auditor appointment",),
    "capital and equity": ("ipo proceeds",),
}

AGMS_TAG_ALIASES: dict[str, str] = {
    "capital and equity": "capital and equity",
    "capital equity": "capital and equity",
    "debt and credit": "debt and credit",
    "debt credit": "debt and credit",
    "acquisition and investment": "acquisition and investment",
    "acquisition investment": "acquisition and investment",
    "board and management": "board and management",
    "board management": "board and management",
    "financial reporting and profit allocation": "financial reporting and profit allocation",
    "financial reporting": "financial reporting and profit allocation",
    "corporate governance": "corporate governance",
    "administrative and other": "administrative and other",
    "administrative other": "administrative and other",
}


@dataclass
class MeetingSummarizer:
    settings: Settings

    def __post_init__(self) -> None:
        return

    def summarize(
        self,
        transcript: str,
        doc_type: SummaryDocType = "agms",
        source_name: str | None = None,
        symbol_override: str | None = None,
    ) -> str:
        normalized_doc_type = str(doc_type).strip().lower()
        if normalized_doc_type not in {"agms", "pubex"}:
            raise RuntimeError("doc_type must be either 'agms' or 'pubex'.")

        if normalized_doc_type == "pubex":
            symbol = symbol_override or extract_symbol_from_filename(source_name) or "NOT_STATED.JK"
        else:
            symbol = extract_symbol(transcript)
        meeting_date = extract_meeting_date(transcript, doc_type=normalized_doc_type)
        company_name = extract_company_name(transcript)

        if normalized_doc_type == "agms":
            summarize_single = self._summarize_single
            summarize_chunk = self._summarize_chunk
            merge_summaries = self._merge_summaries
            normalize_summary = _normalize_agenda_summary
        else:
            summarize_single = self._summarize_single_pubex
            summarize_chunk = self._summarize_chunk_pubex
            merge_summaries = self._merge_summaries_pubex
            normalize_summary = _normalize_pubex_summary

        if len(transcript) <= self.settings.chunk_chars:
            summary = summarize_single(transcript, company_name)
            clean_summary = normalize_summary(summary)
            payload: dict[str, object] = {
                "symbol": symbol,
                "date": meeting_date,
                "summary": clean_summary,
            }
            if normalized_doc_type == "agms":
                payload["tags"] = self._extract_agms_tags(clean_summary, company_name)
            return json.dumps(
                payload,
                ensure_ascii=False,
            )

        chunks = split_text(transcript, self.settings.chunk_chars)
        partials: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            partials.append(summarize_chunk(chunk, index, len(chunks)))

        combined = "\n\n".join(
            f"Chunk {i + 1} summary:\n{summary}" for i, summary in enumerate(partials)
        )
        merged = merge_summaries(combined, company_name)
        clean_summary = normalize_summary(merged)
        payload: dict[str, object] = {
            "symbol": symbol,
            "date": meeting_date,
            "summary": clean_summary,
        }
        if normalized_doc_type == "agms":
            payload["tags"] = self._extract_agms_tags(clean_summary, company_name)
        return json.dumps(
            payload,
            ensure_ascii=False,
        )

    def _summarize_single(self, transcript: str, company_name: str) -> str:
        prompt = f"""
Summarize this meeting transcript into agenda-only output for {company_name}.

Return Markdown using exactly this format:
Agenda #1: SUMMARIZE IN 1-2 sentences.
Agenda #2: ...
Agenda #3: ...

Rules:
- Keep each agenda summary to 1-2 sentences.
- Use only facts from the transcript.
- If agenda title/number is not stated, infer a neutral label from discussion.
- If information is missing, say: Not stated.
- Do not add any other sections, bullets, tables, or notes.

Transcript:
{transcript}
""".strip()
        return self._chat(prompt)

    def _summarize_chunk(self, chunk: str, index: int, total: int) -> str:
        prompt = f"""
You are summarizing chunk {index} of {total} of a long meeting transcript.
Extract agenda-related facts only.

Return concise Markdown with this format only:
Agenda candidates:
- Agenda: <title or inferred label>; Summary: <1-2 sentences with facts>; Evidence: <short factual note>

Rules:
- Include only information explicitly present in this chunk.
- If owner/date/decision is not stated, write Not stated.
- Do not add extra sections.

Chunk text:
{chunk}
""".strip()
        return self._chat(prompt)

    def _merge_summaries(self, partial_summaries: str, company_name: str) -> str:
        prompt = f"""
Merge these chunk-level meeting summaries into one final agenda-only report for {company_name}.
Deduplicate repeated items.

Return Markdown using exactly this format:
Agenda #1: SUMMARIZE IN 1-2 sentences.
Agenda #2: ...
Agenda #3: ...

Rules:
- Keep each agenda line to 1-2 sentences.
- Preserve factual accuracy from chunk summaries only.
- If an agenda is missing details, use Not stated.
- Do not output any extra headings, bullets, tables, or commentary.

Chunk summaries:
{partial_summaries}
""".strip()
        return self._chat(prompt)

    def _summarize_single_pubex(self, transcript: str, company_name: str) -> str:
        prompt = f"""
Summarize this Public Expose transcript into question-and-answer output for {company_name}.

Focus on sections such as:
- Risalah Pertanyaan dan Jawaban
- Ringkasan Pertanyaan dan Jawaban
- Pertanyaan / Question and Jawaban / Answer

Return Markdown using exactly this format:
Q&A #1: Q: <short question summary>. A: <short answer summary or Not stated>.
Q&A #2: Q: ... A: ...
Q&A #3: Q: ... A: ...

Rules:
- Include only factual Q&A content from the transcript.
- If a question has no answer in the transcript, set answer to Not stated.
- Keep each question and answer concise (max 1 sentence each).
- Deduplicate repeated questions.
- Do not add extra headings, bullets, tables, or commentary.

Transcript:
{transcript}
""".strip()
        return self._chat(prompt)

    def _summarize_chunk_pubex(self, chunk: str, index: int, total: int) -> str:
        prompt = f"""
You are summarizing chunk {index} of {total} of a long Public Expose transcript.
Extract question-and-answer facts only.

Return concise Markdown with this format only:
Q&A candidates:
- Q: <question summary or Not stated>; A: <answer summary or Not stated>; Evidence: <short factual note>

Rules:
- Include only information explicitly present in this chunk.
- If only question text exists, set answer to Not stated.
- If only answer text exists without explicit question, set question to Not stated.
- Do not add extra sections.

Chunk text:
{chunk}
""".strip()
        return self._chat(prompt)

    def _merge_summaries_pubex(self, partial_summaries: str, company_name: str) -> str:
        prompt = f"""
Merge these chunk-level Public Expose Q&A summaries into one final Q&A report for {company_name}.
Deduplicate repeated items and preserve factual wording.

Return Markdown using exactly this format:
Q&A #1: Q: <short question summary>. A: <short answer summary or Not stated>.
Q&A #2: Q: ... A: ...
Q&A #3: Q: ... A: ...

Rules:
- Keep each question and answer concise (max 1 sentence each).
- Preserve factual accuracy from chunk summaries only.
- If answer detail is missing, use Not stated.
- Do not output extra headings, bullets, tables, or commentary.

        Chunk summaries:
        {partial_summaries}
""".strip()
        return self._chat(prompt)

    def _extract_agms_tags(self, clean_summary: str, company_name: str) -> list[str]:
        if not clean_summary.strip():
            return []

        taxonomy = "\n".join(f"- {tag}" for tag in AGMS_TAG_KEYWORDS)
        prompt = f"""
Classify this AGM summary of {company_name} into the allowed taxonomy tags below.

Return ONLY a JSON array of strings.

Rules:
- Choose only from the allowed tags below.
- Use English.
- Use lowercase.
- Deduplicate tags.
- Keep only tags clearly supported by the summary.
- Do not include explanations.

Allowed tags:
{taxonomy}

AGM summary:
{clean_summary}
""".strip()

        keyword_tags = _classify_agms_tags(clean_summary)
        llm_tags = _normalize_agms_tags(self._chat(prompt))

        merged: list[str] = []
        seen: set[str] = set()
        for tag in [*keyword_tags, *llm_tags]:
            if tag in seen:
                continue
            seen.add(tag)
            merged.append(tag)
        return merged

    def _chat(self, prompt: str) -> str:
        if self.settings.api == "openai":
            return self._chat_openai(prompt)
        return self._chat_gemini(prompt)

    def _chat_openai(self, prompt: str) -> str:
        endpoint = self.settings.base_url or "https://api.openai.com/v1/chat/completions"

        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }

        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=240) as res:
                body = res.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

        data = json.loads(body)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("LLM returned a blank response.")

        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            output = content
        elif isinstance(content, list):
            texts = [part.get("text", "") for part in content if isinstance(part, dict)]
            output = "\n".join(texts)
        else:
            output = ""

        output = output.strip()
        if not output:
            raise RuntimeError("LLM returned a blank response.")
        return output

    def _chat_gemini(self, prompt: str) -> str:
        endpoint = self.settings.base_url or (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.settings.model}:generateContent"
        )

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.settings.temperature},
        }
        query = parse.urlencode({"key": self.settings.api_key})
        url = f"{endpoint}?{query}" if "?" not in endpoint else f"{endpoint}&{query}"

        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=240) as res:
                body = res.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API request failed: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc

        data = json.loads(body)
        texts: list[str] = []
        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text_part = part.get("text")
                if text_part:
                    texts.append(text_part)

        output = "\n".join(texts)

        output = output.strip()
        if not output:
            raise RuntimeError("LLM returned a blank response.")
        return output


def split_text(text: str, max_chars: int) -> list[str]:
    if max_chars < 1000:
        raise RuntimeError("SUMMARY_CHUNK_CHARS must be at least 1000.")

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = para.strip()
        if not candidate:
            continue

        if len(candidate) > max_chars:
            # Fallback for extremely long paragraphs.
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(candidate), max_chars):
                chunks.append(candidate[i : i + max_chars])
            continue

        tentative = f"{current}\n\n{candidate}".strip() if current else candidate
        if len(tentative) <= max_chars:
            current = tentative
        else:
            if current:
                chunks.append(current)
            current = candidate

    if current:
        chunks.append(current)

    if not chunks:
        raise RuntimeError("No valid text chunks were generated from transcript.")

    return chunks


def extract_company_name(transcript: str) -> str:
    # Search near the top first, where company identity is usually printed in AGM docs.
    preview = transcript[:4000]

    patterns = [
        r"\bPT\s+[A-Z][A-Za-z0-9&.,'()\-/ ]{2,120}\b(?:Tbk|TBK)\b",
        r"\bPT\s+[A-Z][A-Za-z0-9&.,'()\-/ ]{2,120}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, preview)
        if match:
            return " ".join(match.group(0).split())

    return "PT [Not stated]"


def extract_symbol(transcript: str) -> str:
    preview = transcript[:6000]
    stopwords = {
        "STOCK",
        "CODE",
        "KODE",
        "EMITEN",
        "ISSUER",
        "SAHAM",
        "SUM",
        "OF",
        "ATTENDED",
        "JUMLAH",
        "KEHADIRAN",
        "REPORT",
        "GENERATED",
        "NAMA",
    }

    patterns = [
        r"\bKode\s+Emiten\s*[:\-]?\s*([A-Za-z]{2,10})\b",
        r"\bIssuer\s+Code\s*[:\-]?\s*([A-Za-z]{2,10})\b",
        r"\bKode\s+Saham\s*[:\-]?\s*([A-Za-z]{2,10})\b",
        r"\bStock\s+Code\s*[:\-]?\s*([A-Za-z]{2,10})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, preview, flags=re.IGNORECASE)
        if match:
            raw_symbol = match.group(1).strip().upper().rstrip(".")
            raw_symbol = raw_symbol.removesuffix(".JK")
            if raw_symbol in stopwords or not raw_symbol.isalpha() or len(raw_symbol) > 6:
                continue
            return f"{raw_symbol}.JK"

    stock_code_window = re.search(r"stock\s+code[\s\S]{0,180}", preview, flags=re.IGNORECASE)
    if stock_code_window:
        for candidate in re.findall(r"\b([A-Z]{2,6})\b", stock_code_window.group(0)):
            if candidate not in stopwords:
                return f"{candidate}.JK"

    return "NOT_STATED.JK"


def extract_symbol_from_filename(source_name: str | None) -> str | None:
    if not source_name:
        return None

    normalized_name = source_name.strip().replace("\\", "/")
    base_name = normalized_name.rsplit("/", maxsplit=1)[-1]
    stem = base_name.rsplit(".", maxsplit=1)[0]

    # Expected PubEx filename shape: YYYYMMDD_SYMBOL_...
    match = re.match(r"^\d{8}_([A-Za-z]{4})(?:_|$)", stem)
    if not match:
        return None

    return f"{match.group(1).upper()}.JK"


def extract_meeting_date(transcript: str, doc_type: SummaryDocType = "agms") -> str:
    preview = transcript[:16000]
    primary_candidates: list[str] = []
    generic_candidates: list[str] = []

    normalized_doc_type = str(doc_type).strip().lower()
    if normalized_doc_type == "pubex":
        patterns = [
            r"public\s+expose(?:\s+tahunan)?[\s\S]{0,260}?pada\s+hari\s+[A-Za-z]+,\s*([^,.;]+)",
            r"public\s+expose(?:\s+tahunan)?[\s\S]{0,260}?pada\s+tanggal\s+([^,.;]+)",
            r"dilaksanakan\s+pada\s+hari\s+[A-Za-z]+,\s*([^,.;]+)",
            r"dilaksanakan\s+pada\s+tanggal\s+([^,.;]+)",
            r"report\s+generated\s+([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
        ]
    else:
        patterns = [
            r"penyelenggaraan\s+Rapat\s+Umum\s+Pemegang\s+Saham[\s\S]{0,220}?dilaksanakan\s+pada\s+tanggal\s+([^,.;]+)",
            r"Rapat\s+Umum\s+Pemegang\s+Saham[\s\S]{0,220}?dilaksanakan\s+pada\s+tanggal\s+([^,.;]+)",
            r"hasil\s+penyelenggaraan[\s\S]{0,260}?dilaksanakan\s+pada\s+tanggal\s+([^,.;]+)",
            r"general\s+meeting\s+of\s+shareholder(?:'s|s)?\s+result\s+on\s+([^,.;]+)",
            r"dilaksanakan\s+pada\s+tanggal\s+([^,.;]+)",
        ]

    generic_patterns = [
        r"\b([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})\b",
        r"\b(?:senin|selasa|rabu|kamis|jumat|jum'at|sabtu|minggu|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*,\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
        r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, preview, flags=re.IGNORECASE):
            primary_candidates.append(match.group(1).strip())

    for pattern in generic_patterns:
        for match in re.finditer(pattern, preview, flags=re.IGNORECASE):
            generic_candidates.append(match.group(1).strip())

    for raw in [*primary_candidates, *generic_candidates]:
        parsed = _parse_iso_date(raw)
        if parsed:
            return parsed

    return "Not stated"


def _parse_iso_date(raw: str) -> str | None:
    cleaned = " ".join(raw.split())
    cleaned = re.sub(r"\b(pukul|jam)\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(
        r"^(senin|selasa|rabu|kamis|jumat|jum'at|sabtu|minggu|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*,?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not cleaned:
        return None

    numeric_match = re.search(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", cleaned)
    if numeric_match:
        normalized = numeric_match.group(1).replace(".", "-").replace("/", "-")
        for fmt in ("%d-%m-%Y", "%d-%m-%y"):
            try:
                parsed = datetime.strptime(normalized, fmt)
                if _is_plausible_meeting_year(parsed.year):
                    return parsed.strftime("%Y-%m-%d")
            except ValueError:
                pass

    match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", cleaned)
    if not match:
        return None

    day = int(match.group(1))
    month_raw = match.group(2).lower().strip(".")
    year = int(match.group(3))

    month_map = {
        "jan": 1,
        "januari": 1,
        "january": 1,
        "feb": 2,
        "februari": 2,
        "february": 2,
        "mar": 3,
        "maret": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "mei": 5,
        "may": 5,
        "jun": 6,
        "juni": 6,
        "june": 6,
        "jul": 7,
        "juli": 7,
        "july": 7,
        "agu": 8,
        "agt": 8,
        "agustus": 8,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "okt": 10,
        "oktober": 10,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "des": 12,
        "desember": 12,
        "dec": 12,
        "december": 12,
    }
    month = month_map.get(month_raw)
    if month is None:
        return None

    try:
        parsed = datetime(year, month, day)
        if not _is_plausible_meeting_year(parsed.year):
            return None
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _is_plausible_meeting_year(year: int) -> bool:
    current_year = datetime.now().year
    return 1990 <= year <= current_year + 2


def _normalize_agenda_summary(summary: str) -> str:
    body = summary.strip()
    if not body:
        return ""

    if body.startswith("```"):
        body = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)

    lines = [re.sub(r"^-+\s*", "", line).strip() for line in body.splitlines() if line.strip()]
    non_title_lines = [line for line in lines if not line.upper().startswith("AGMS SUMMARY OF")]

    agenda_lines = [
        line
        for line in non_title_lines
        if re.match(r"^Agenda\s*#?\d+\s*:", line, flags=re.IGNORECASE)
    ]

    if agenda_lines:
        return "\n".join(agenda_lines)

    return "\n".join(non_title_lines)


def _normalize_pubex_summary(summary: str) -> str:
    body = summary.strip()
    if not body:
        return ""

    if body.startswith("```"):
        body = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)

    lines = [re.sub(r"^-+\s*", "", line).strip() for line in body.splitlines() if line.strip()]
    non_title_lines = [
        line
        for line in lines
        if not line.upper().startswith("PUBEX SUMMARY OF")
        and not line.upper().startswith("PUBLIC EXPOSE SUMMARY OF")
    ]

    qa_lines = [
        line
        for line in non_title_lines
        if re.match(r"^(Q\s*&\s*A|Q/A|QA)\s*#?\d+\s*:", line, flags=re.IGNORECASE)
    ]

    if qa_lines:
        return "\n".join(qa_lines)

    return "\n".join(non_title_lines)


def _normalize_tags(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []

    candidates: list[str] = []

    parsed: object | None = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                candidates.append(item)
    elif isinstance(parsed, dict):
        for key in ("tags", "tag"):
            value = parsed.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        candidates.append(item)
                break
    else:
        normalized_fallback = text
        if normalized_fallback.startswith("```"):
            normalized_fallback = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", normalized_fallback)
            normalized_fallback = re.sub(r"\s*```$", "", normalized_fallback)
        normalized_fallback = normalized_fallback.replace(";", ",")
        for line in normalized_fallback.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            stripped = re.sub(r"^[-*•\d.)\s]+", "", stripped).strip()
            candidates.extend(part.strip() for part in stripped.split(",") if part.strip())

    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        tag = candidate.strip().lower()
        tag = tag.strip("\"'`")
        tag = tag.replace("/", " ").replace("_", " ")
        tag = re.sub(r"\s+", " ", tag)
        tag = re.sub(r"[^a-z0-9 \-]", "", tag).strip(" -")
        if not tag or tag in {"not stated", "n/a", "na", "none"}:
            continue

        words = tag.split()
        if len(words) > 2:
            tag = " ".join(words[:2])
        if not tag:
            continue

        if tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag)
        if len(cleaned) >= 8:
            break

    return cleaned


def _normalize_agms_tags(raw: str) -> list[str]:
    normalized = _normalize_tags(raw)
    cleaned: list[str] = []
    seen: set[str] = set()
    for tag in normalized:
        canonical = _canonicalize_agms_tag(tag)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        cleaned.append(canonical)
    return cleaned


def _canonicalize_agms_tag(raw_tag: str) -> str | None:
    tag = raw_tag.strip().lower()
    tag = tag.strip("\"'`[](){}")
    tag = re.sub(r"[^a-z0-9&\-\s]", " ", tag)
    tag = tag.replace("&", " and ")
    tag = re.sub(r"\s+", " ", tag).strip()
    if not tag:
        return None

    if tag in AGMS_TAG_KEYWORDS:
        return tag

    if tag in AGMS_TAG_ALIASES:
        return AGMS_TAG_ALIASES[tag]

    for canonical, keywords in AGMS_TAG_KEYWORDS.items():
        if tag in keywords:
            return canonical

    for canonical, signals in AGMS_EXTRA_CLASSIFICATION_SIGNALS.items():
        if tag in signals:
            return canonical
    return None


def _classify_agms_tags(clean_summary: str) -> list[str]:
    text = clean_summary.lower()
    matched: list[str] = []
    seen: set[str] = set()

    for canonical, keywords in AGMS_TAG_KEYWORDS.items():
        if any(keyword in text for keyword in keywords) and canonical not in seen:
            seen.add(canonical)
            matched.append(canonical)

    for canonical, signals in AGMS_EXTRA_CLASSIFICATION_SIGNALS.items():
        if any(signal in text for signal in signals) and canonical not in seen:
            seen.add(canonical)
            matched.append(canonical)

    return matched
