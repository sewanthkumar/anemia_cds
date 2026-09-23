"""
report_parser.py — Extract CBC values from PDF lab reports or images.
Uses PyMuPDF for PDF text, OpenCV + Tesseract OCR for images,
and LLMClient for intelligent extraction (never calls LLM backends directly).
"""
import os
import re
import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
from rich.console import Console
from modules.llm_client import LLMClient
import config

console = Console()

CBC_KEYS = ['Hb', 'MCV', 'MCH', 'MCHC', 'RBC', 'RDW', 'Hematocrit']

PDF_PROMPT_TEMPLATE = (
    "Extract CBC values from this lab report text. "
    "Return ONLY a JSON object with exactly these keys: "
    "Hb, MCV, MCH, MCHC, RBC, RDW, Hematocrit. "
    "Use null for missing values. Ignore units. Text: {text}"
)

VISION_PROMPT = (
    "This is a medical CBC lab report. Extract these values and return ONLY JSON: "
    "{Hb, MCV, MCH, MCHC, RBC, RDW, Hematocrit}. Use null for missing. Numbers only, no units."
)


class ReportParser:
    """Parse CBC values from PDF reports or blood test images."""

    def __init__(self, client: LLMClient):
        self.client = client

    # ────────────────────────────────── PDF ──────────────────────────────────

    def parse_pdf(self, path: str) -> dict:
        """
        Extract CBC values from a PDF lab report.
        Step 1: Extract all text with PyMuPDF.
        Step 2: Send text to LLM → parse JSON response.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")

        console.print(f"[cyan]📄 Parsing PDF: {os.path.basename(path)}[/cyan]")

        # Extract text
        doc = fitz.open(path)
        text = ' '.join([page.get_text() for page in doc])
        doc.close()

        if not text.strip():
            raise ValueError("PDF text extraction returned empty. File may be scanned — try image input instead.")

        console.print(f"   Extracted {len(text)} characters. Sending to LLM...")

        prompt = PDF_PROMPT_TEMPLATE.format(text=text[:4000])  # Cap to avoid token overflow
        data = self.client.chat_json(prompt)
        return self._clean_extracted(data)

    # ────────────────────────────────── IMAGE ────────────────────────────────

    def parse_image(self, path: str) -> dict:
        """
        Extract CBC values from an image (photo of lab report).
        Step 1: Preprocess with OpenCV (grayscale, denoise, threshold).
        Step 2: Try Tesseract OCR.
        Step 3: If OCR output < 80 chars or no numbers found → use vision model.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        console.print(f"[cyan]🖼 Parsing image: {os.path.basename(path)}[/cyan]")

        # Step 1: Preprocess
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")

        preprocessed_path = self._preprocess_image(img, path)

        # Step 2: Tesseract OCR
        ocr_text = pytesseract.image_to_string(preprocessed_path)
        console.print(f"   OCR extracted {len(ocr_text)} characters.")

        has_numbers = bool(re.search(r'\d+\.?\d*', ocr_text))

        if len(ocr_text.strip()) >= 80 and has_numbers:
            console.print("   OCR quality: [green]good[/green] — using text extraction.")
            prompt = PDF_PROMPT_TEMPLATE.format(text=ocr_text)
            data = self.client.chat_json(prompt)
        else:
            console.print("   OCR quality: [yellow]poor[/yellow] — switching to vision model.")
            data = self.client.vision(path, VISION_PROMPT)
            if isinstance(data, str):
                import json, re as _re
                match = _re.search(r'\{.*\}', data, _re.DOTALL)
                if match:
                    data = json.loads(match.group())
                else:
                    raise ValueError(f"Vision model returned non-JSON: {data[:200]}")

        # Cleanup temp preprocessed file
        if os.path.exists(preprocessed_path) and preprocessed_path != path:
            os.remove(preprocessed_path)

        return self._clean_extracted(data)

    def _preprocess_image(self, img: np.ndarray, original_path: str) -> str:
        """Apply OpenCV preprocessing and save to uploads/. Returns path to preprocessed file."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        os.makedirs(config.UPLOAD_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(original_path))[0]
        out_path = os.path.join(config.UPLOAD_DIR, f'{base}_preprocessed.png')
        cv2.imwrite(out_path, thresh)
        return out_path

    def _clean_extracted(self, data: dict) -> dict:
        """
        Validate and clean extracted CBC values.
        Convert strings to floats, handle nulls, flag missing.
        """
        cleaned = {}
        for key in CBC_KEYS:
            val = data.get(key) or data.get(key.lower())
            if val is None or val == 'null':
                cleaned[key] = None
                console.print(f"   [yellow]{key}: MISSING (null returned by LLM)[/yellow]")
            else:
                try:
                    cleaned[key] = float(val)
                except (ValueError, TypeError):
                    cleaned[key] = None
                    console.print(f"   [yellow]{key}: Could not parse '{val}'[/yellow]")

        present = sum(1 for v in cleaned.values() if v is not None)
        console.print(f"   Extracted {present}/{len(CBC_KEYS)} CBC values.")

        if present < 5:
            raise ValueError(
                f"Extraction failed — only {present} values found. "
                "Check that the file is a readable CBC lab report."
            )

        return cleaned
