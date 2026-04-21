from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def extract_text_from_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise RuntimeError(f"PDF not found: {pdf_path}")

    text = _extract_text_with_pdftotext(pdf_path)
    if text:
        return text

    text = _extract_text_with_ocr(pdf_path)
    if text:
        return text

    raise RuntimeError(
        "Text extraction failed. For scanned PDFs, ensure pdftoppm (poppler) and tesseract are installed and available in PATH."
    )


def _extract_text_with_pdftotext(pdf_path: Path) -> str:
    cmd = ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"]
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
            "pdftotext not found in PATH. Install poppler-utils and ensure pdftotext is available."
        ) from exc

    if result.returncode != 0:
        error_message = (result.stderr or "").strip() or "Unknown pdftotext error"
        raise RuntimeError(f"Failed to extract text from PDF: {error_message}")

    return (result.stdout or "").replace("\x0c", "").strip()


def _extract_text_with_ocr(pdf_path: Path) -> str:
    engine = os.getenv("OCR_ENGINE", "auto").strip().lower()
    if engine not in {"auto", "ocrmypdf", "tesseract"}:
        raise RuntimeError("OCR_ENGINE must be one of: auto, ocrmypdf, tesseract.")

    errors: list[str] = []

    if engine in {"auto", "ocrmypdf"}:
        try:
            text = _extract_text_with_ocrmypdf(pdf_path)
            if text:
                return text
        except RuntimeError as exc:
            errors.append(str(exc))
            if engine == "ocrmypdf":
                raise

    if engine in {"auto", "tesseract"}:
        try:
            text = _extract_text_with_tesseract_pipeline(pdf_path)
            if text:
                return text
        except RuntimeError as exc:
            errors.append(str(exc))
            if engine == "tesseract":
                raise

    if errors:
        raise RuntimeError("OCR did not produce text. Details: " + " | ".join(errors))

    return ""


def _extract_text_with_ocrmypdf(pdf_path: Path) -> str:
    ocr_lang = os.getenv("OCR_LANG", "eng")

    with tempfile.TemporaryDirectory(prefix="meeting_summarizer_") as temp_dir:
        temp_path = Path(temp_dir)
        sidecar_path = temp_path / "ocr.txt"
        output_pdf_path = temp_path / "ocr.pdf"

        cmd = [
            "ocrmypdf",
            "--skip-text",
            "--sidecar",
            str(sidecar_path),
            "-l",
            ocr_lang,
            str(pdf_path),
            str(output_pdf_path),
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
                "ocrmypdf not found in PATH. Install OCRmyPDF and ensure ocrmypdf is available."
            ) from exc

        if result.returncode != 0:
            error_message = (result.stderr or "").strip() or "Unknown ocrmypdf error"
            raise RuntimeError(f"OCRmyPDF failed: {error_message}")

        if not sidecar_path.exists():
            return ""

        return sidecar_path.read_text(encoding="utf-8", errors="ignore").strip()


def _extract_text_with_tesseract_pipeline(pdf_path: Path) -> str:
    ocr_lang = os.getenv("OCR_LANG", "eng")

    with tempfile.TemporaryDirectory(prefix="meeting_summarizer_") as temp_dir:
        image_prefix = Path(temp_dir) / "page"
        render_cmd = ["pdftoppm", "-png", str(pdf_path), str(image_prefix)]

        try:
            render_result = subprocess.run(
                render_cmd,
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

        if render_result.returncode != 0:
            error_message = (render_result.stderr or "").strip() or "Unknown pdftoppm error"
            raise RuntimeError(f"Failed to render PDF pages for OCR: {error_message}")

        images = sorted(Path(temp_dir).glob("page-*.png"))
        if not images:
            return ""

        page_texts: list[str] = []
        for image_path in images:
            ocr_cmd = ["tesseract", str(image_path), "stdout", "-l", ocr_lang]
            try:
                ocr_result = subprocess.run(
                    ocr_cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "tesseract not found in PATH. Install Tesseract OCR and ensure tesseract is available."
                ) from exc

            if ocr_result.returncode != 0:
                error_message = (ocr_result.stderr or "").strip() or "Unknown tesseract error"
                raise RuntimeError(f"OCR failed on {image_path.name}: {error_message}")

            page_text = (ocr_result.stdout or "").strip()
            if page_text:
                page_texts.append(page_text)

        return "\n\n".join(page_texts).strip()
