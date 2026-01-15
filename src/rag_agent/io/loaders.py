"""Unified PDF processing, description and loading utilities.

This module provides comprehensive PDF document processing including:
- Text extraction from PDFs
- Image extraction and AI-based description
- Word document to PDF conversion
- Document loading for RAG pipeline

Dependencies:
    pymupdf, mistralai, python-dotenv, PyPDF2
"""
from __future__ import annotations

# Standard library
import base64
import concurrent.futures as cf
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Third-party
import fitz  # PyMuPDF
from PyPDF2 import PdfReader

try:
    from mistralai import Mistral
except ImportError as _err:
    raise ImportError(
        "The 'mistralai' library is required. Install with: pip install mistralai"
    ) from _err

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Optional: dotenv not required

# Module logger
logger = logging.getLogger(__name__)

# MIME type utilities
_MIME_BY_EXT = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "bmp": "image/bmp",
    "gif": "image/gif",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "webp": "image/webp",
}


def _guess_mime(ext: str) -> str:
    """Return a MIME type for the given file extension.
    
    Args:
        ext: File extension (without dot)
        
    Returns:
        MIME type string
    """
    return _MIME_BY_EXT.get(ext.lower(), f"image/{ext.lower()}")


# PDF extraction (text + images)

@dataclass
class PDFImage:
    """Represents an image extracted from a PDF.
    
    Attributes:
        page_number: Page number where image appears
        index_on_page: Image index on the page
        xref: PDF cross-reference number
        bytes: Raw image data
        extension: Image file extension
        width: Image width in pixels
        height: Image height in pixels
    """
    page_number: int
    index_on_page: int
    xref: int
    bytes: bytes
    extension: str
    width: int
    height: int


class PDFProcessor:
    """Extracts native text and raster images from PDF files.
    
    This class uses PyMuPDF (fitz) to process PDF documents and extract
    both textual content and embedded images.
    """

    def __init__(self, pdf_path: str | Path):
        """Initialize PDF processor.
        
        Args:
            pdf_path: Path to the PDF file
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.is_file():
            raise FileNotFoundError(self.pdf_path)
        self.doc = fitz.open(self.pdf_path)
        logger.info("Opened PDF: %s (%d pages)", self.pdf_path, len(self.doc))

    def extract_text_per_page(self) -> List[str]:
        """Extract text from each page.
        
        Returns:
            List of text strings, one per page
        """
        texts: List[str] = []
        for idx, page in enumerate(self.doc, 1):
            try:
                texts.append(page.get_text("text", sort=True))
            except Exception as e:
                logger.warning("Error extracting text from page %d: %s", idx, e)
                texts.append("[Text extraction error]")
        return texts

    def extract_images_per_page(
        self,
        *,
        dedup: bool = True,
        min_width: int = 0,
        min_height: int = 0,
    ) -> Dict[int, List[PDFImage]]:
        """Extract images from each page with optional filtering.
        
        Args:
            dedup: If True, deduplicate images by xref across pages
            min_width: Minimum image width in pixels
            min_height: Minimum image height in pixels
            
        Returns:
            Dictionary mapping page number to list of PDFImage objects
        """
        images_by_page: Dict[int, List[PDFImage]] = {}
        seen_xrefs: Set[int] = set()

        for page_number, page in enumerate(self.doc, 1):
            kept: List[PDFImage] = []
            img_idx = 0
            try:
                infos = page.get_image_info(hashes=False, xrefs=True)
            except Exception as e:
                logger.error(
                    "Error getting image info for page %d: %s", page_number, e, exc_info=True
                )
                continue

            for info in infos:
                xref = info.get("xref") or 0
                if xref <= 0:
                    continue
                if dedup and xref in seen_xrefs:
                    continue

                width, height = info.get("width", 0), info.get("height", 0)
                if width < min_width or height < min_height:
                    continue
                try:
                    img_dict = self.doc.extract_image(xref)
                    img_bytes = img_dict["image"]
                    img_ext = img_dict.get("ext", "bin")
                except Exception as e:
                    logger.error("Image extraction failed for xref %d: %s", xref, e)
                    continue

                img_idx += 1
                kept.append(
                    PDFImage(
                        page_number=page_number,
                        index_on_page=img_idx,
                        xref=xref,
                        bytes=img_bytes,
                        extension=img_ext,
                        width=width,
                        height=height,
                    )
                )
                if dedup:
                    seen_xrefs.add(xref)

            if kept:
                images_by_page[page_number] = kept

        return images_by_page


class MistralImageDescriber:
    """Generates concise descriptions for images using Mistral's multimodal chat API.
    
    Uses Mistral's vision capabilities to create objective image captions.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "mistral-small-latest",
        temperature: float = 0.1,
    ) -> None:
        """Initialize the image describer.
        
        Args:
            api_key: Mistral API key
            model: Model name for vision/multimodal tasks
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.prompt = "Describe this image concisely and objectively in English."
        logger.info("MistralImageDescriber initialized with model %s", model)

    def describe(self, img: PDFImage) -> str:
        """Generate a text description for an image.
        
        Args:
            img: PDFImage object containing image data
            
        Returns:
            Text description of the image
        """
        b64_image = base64.b64encode(img.bytes).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:{_guess_mime(img.extension)};base64,{b64_image}",
                    },
                ],
            }
        ]
        try:
            resp = self.client.chat.complete(
                model=self.model, messages=messages, temperature=self.temperature
            )
            if resp.choices:
                return resp.choices[0].message.content.strip()
            return "<no description>"
        except Exception as e:
            logger.error("Mistral API error: %s", e)
            return f"<API error: {type(e).__name__}>"


class PDFDescriber:
    """Handles full text extraction with optional image captioning in parallel.
    
    Orchestrates PDF processing by extracting text and optionally generating
    AI-based descriptions for embedded images.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        *,
        output_path: str | Path | None = None,
        model: str = "mistral-small-latest",
        workers: int = 4,
        dedup: bool = True,
        min_width: int = 0,
        min_height: int = 0,
        process_images: bool = True,
    ) -> None:
        self.processor = PDFProcessor(pdf_path)
        self.process_images = process_images 
        self.describer: Optional[MistralImageDescriber] = None

        if self.process_images:
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                logger.warning(
                    "MISTRAL_API_KEY not defined. Image description disabled for %s.",
                    pdf_path
                )
                self.process_images = False
            else:
                self.describer = MistralImageDescriber(api_key, model=model)
        self.workers = max(1, workers)
        self.dedup = dedup
        self.min_width, self.min_height = min_width, min_height
        self.output_path = Path(output_path) if output_path else None

    def build_description(self) -> str:
        """Build full description with text and optional image captions.
        
        Returns:
            Complete text description with embedded image captions
        """
        texts = self.processor.extract_text_per_page()
        captions: Dict[int, List[Tuple[int, str]]] = {}

        if self.process_images and self.describer:
            images = self.processor.extract_images_per_page(
                dedup=self.dedup, min_width=self.min_width, min_height=self.min_height
            )
            total_imgs = sum(len(v) for v in images.values())


            if total_imgs > 0:
                flat: List[PDFImage] = [img for lst in images.values() for img in lst]
                with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
                    fut_to_img = {ex.submit(self.describer.describe, i): i for i in flat}
                    for fut in cf.as_completed(fut_to_img):
                        img = fut_to_img[fut]
                        try:
                            cap = fut.result()
                        except Exception as e:
                            cap = f"<thread error: {e}>"
                        captions.setdefault(img.page_number, []).append((img.index_on_page, cap))

        # Assemble final output
        out: List[str] = []
        num_pages = len(texts)
        for p_idx, txt in enumerate(texts, 1):
            out.append(f"=== Page {p_idx}/{num_pages} ===\n")
            out.append(txt.strip() or "[No text extracted]")
            out.append("")
            if p_idx in captions:
                out.append("-- Images on this page --")
                for idx, cap in sorted(captions[p_idx]):
                    out.append(f"[Image {idx}] {cap}")
                out.append("")
            if p_idx < num_pages:
                out.append("-" * 70 + "\n")
        full = "\n".join(out)
        if self.output_path:
            self.output_path.write_text(full, encoding="utf-8")
        return full


def run_describer(
    *,
    pdf_path: Path,
    api_key: str | None = None,
    min_w: int = 0,
    min_h: int = 0,
    model: str = "mistral-small-latest",
    process_images: bool = True,
) -> List[Tuple[str, int]]:
    """Convenience wrapper returning (text_with_captions, page_number)."""

    # Preserve env variable behaviour but allow override for convenience
    if api_key:
        os.environ.setdefault("MISTRAL_API_KEY_mathis", api_key)

    describer = PDFDescriber(
        pdf_path=pdf_path,
        model=model,
        workers=4,
        dedup=True,
        min_width=min_w,
        min_height=min_h,
        process_images=process_images,
    )
    raw = describer.build_description()

    pages: List[Tuple[str, int]] = []
    for bloc in raw.split("=== Page ")[1:]:
        header, *content = bloc.split("\n", 1)
        page_num = int(header.split("/")[0])
        page_txt = content[0].strip() if content else ""
        pages.append((page_txt, page_num))
    return pages


# Import kept here to avoid circular imports
from .converters import DocumentConverter  # noqa: E402


class DocumentLoader:
    """Recursively converts Word docs to PDF then extracts per-page enriched text."""

    def __init__(
        self,
        *,
        data_dir: Path,
        pdf_output_dir: Path,
        converter: DocumentConverter,
        processed_files: Optional[Set[str]] = None,
        process_images: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.pdf_output_dir = pdf_output_dir
        self.converter = converter
        self.processed_files = processed_files or set()
        self.process_images = process_images
        self.pdf_output_dir.mkdir(parents=True, exist_ok=True)

    def read_pdf_pages(self, pdf_path: Path) -> List[str]:
        """Extract text from PDF pages using PyPDF2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text strings, one per page
        """
        reader = PdfReader(str(pdf_path))
        return [page.extract_text() or "" for page in reader.pages]

    def load(self) -> List[Tuple[str, str, int]]:
        """Return [(page_text, pdf_filename, page_number), ...] for *new* docs."""
        docs: List[Tuple[str, str, int]] = []
        new_files = 0

        print(f"🕵️  Searching for documents in folder: {self.data_dir.resolve()}")

        for file in self.data_dir.rglob("*.[dD][oO][cC]*"):
            if file.name in self.processed_files:
                continue
            try:
                pdf = self.converter.to_pdf(file)
                new_files += 1
                self.processed_files.add(file.name)

                for page_text, page_num in run_describer(
                    pdf_path=pdf, min_w=50, min_h=50, process_images=self.process_images
                ):
                    docs.append((page_text, pdf.name, page_num))
            except Exception as e:
                logger.error("%s ignored (%s)", file.name, e)

        logger.info("✅ %d new files processed, %d pages extracted", new_files, len(docs))
        return docs
