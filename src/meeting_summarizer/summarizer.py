from __future__ import annotations

import json
import re
from datetime import datetime
from urllib import parse, request
from urllib.error import HTTPError, URLError
from dataclasses import dataclass

from .config import Settings

SYSTEM_PROMPT = """You are an expert meeting analyst.
Return clean text.
Be factual and only use information found in the transcript.
If information is missing, explicitly say it is not stated.
"""


@dataclass
class MeetingSummarizer:
    settings: Settings

    def __post_init__(self) -> None:
        return

    def summarize(self, transcript: str) -> str:
        symbol = extract_symbol(transcript)
        meeting_date = extract_meeting_date(transcript)
        company_name = extract_company_name(transcript)

        if len(transcript) <= self.settings.chunk_chars:
            summary = self._summarize_single(transcript, company_name)
            clean_summary = _normalize_agenda_summary(summary)
            return json.dumps(
                {
                    "symbol": symbol,
                    "date": meeting_date,
                    "summary": clean_summary,
                },
                ensure_ascii=False,
            )

        chunks = split_text(transcript, self.settings.chunk_chars)
        partials: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            partials.append(self._summarize_chunk(chunk, index, len(chunks)))

        combined = "\n\n".join(
            f"Chunk {i + 1} summary:\n{summary}" for i, summary in enumerate(partials)
        )
        merged = self._merge_summaries(combined, company_name)
        clean_summary = _normalize_agenda_summary(merged)
        return json.dumps(
            {
                "symbol": symbol,
                "date": meeting_date,
                "summary": clean_summary,
            },
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
            with request.urlopen(req, timeout=120) as res:
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
            with request.urlopen(req, timeout=120) as res:
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
    preview = transcript[:4000]
    patterns = [
        r"\bKode\s+Emiten\s*[:\-]?\s*([A-Za-z]{2,10})\b",
        r"\bIssuer\s+Code\s*[:\-]?\s*([A-Za-z]{2,10})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, preview, flags=re.IGNORECASE)
        if match:
            raw_symbol = match.group(1).strip().upper()
            if raw_symbol.endswith(".JK"):
                return raw_symbol
            return f"{raw_symbol}.JK"

    return "NOT_STATED.JK"


def extract_meeting_date(transcript: str) -> str:
    preview = transcript[:12000]
    candidates: list[str] = []
    patterns = [
        r"penyelenggaraan\s+Rapat\s+Umum\s+Pemegang\s+Saham[\s\S]{0,220}?dilaksanakan\s+pada\s+tanggal\s+([^,.;\n]+)",
        r"Rapat\s+Umum\s+Pemegang\s+Saham[\s\S]{0,220}?dilaksanakan\s+pada\s+tanggal\s+([^,.;\n]+)",
        r"hasil\s+penyelenggaraan[\s\S]{0,260}?dilaksanakan\s+pada\s+tanggal\s+([^,.;\n]+)",
        r"general\s+meeting\s+of\s+shareholder(?:'s|s)?\s+result\s+on\s+([^,.;\n]+)",
        r"dilaksanakan\s+pada\s+tanggal\s+([^,.;\n]+)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, preview, flags=re.IGNORECASE):
            candidates.append(match.group(1).strip())

    for raw in candidates:
        parsed = _parse_iso_date(raw)
        if parsed:
            return parsed

    return "Not stated"


def _parse_iso_date(raw: str) -> str | None:
    cleaned = " ".join(raw.split())
    cleaned = re.sub(r"\b(pukul|jam)\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    if not cleaned:
        return None

    normalized = cleaned.replace(".", "-").replace("/", "-")
    first_token = normalized.split()[0]

    for fmt in ("%d-%m-%Y", "%d-%m-%y"):
        try:
            return datetime.strptime(first_token, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    match = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", cleaned)
    if not match:
        return None

    day = int(match.group(1))
    month_raw = match.group(2).lower()
    year = int(match.group(3))

    month_map = {
        "januari": 1,
        "februari": 2,
        "maret": 3,
        "april": 4,
        "mei": 5,
        "juni": 6,
        "juli": 7,
        "agustus": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "desember": 12,
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    month = month_map.get(month_raw)
    if month is None:
        return None

    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _normalize_agenda_summary(summary: str) -> str:
    body = summary.strip()
    if not body:
        return ""

    if body.startswith("```"):
        body = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)

    lines = [line.strip() for line in body.splitlines() if line.strip()]
    non_title_lines = [line for line in lines if not line.upper().startswith("AGMS SUMMARY OF")]

    agenda_lines = [
        line
        for line in non_title_lines
        if re.match(r"^Agenda\s*#?\d+\s*:", line, flags=re.IGNORECASE)
    ]

    if agenda_lines:
        return "\n".join(agenda_lines)

    return "\n".join(non_title_lines)
