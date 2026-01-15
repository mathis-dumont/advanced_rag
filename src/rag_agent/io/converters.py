"""Document conversion utilities.

This module handles conversion of Word documents (.doc/.docx) to PDF format
using LibreOffice's headless mode.
"""
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentConverter:
    """Converts Word documents to PDF using LibreOffice."""
    
    def __init__(self, out_dir: Path):
        """Initialize the document converter.
        
        Args:
            out_dir: Output directory for converted PDF files
            
        Raises:
            RuntimeError: If LibreOffice (soffice) is not found in PATH
        """
        self.out_dir = out_dir
        
        self.soffice = shutil.which("soffice")
        if not self.soffice:
            raise RuntimeError(
                "LibreOffice (soffice) not found in PATH. "
                "Please install LibreOffice and ensure 'soffice' is in your PATH."
            )
        logger.info("DocumentConverter initialized with LibreOffice at: %s", self.soffice)

    def to_pdf(self, doc_path: Path) -> Path:
        """Convert a .doc/.docx file to PDF using LibreOffice headless mode.
        
        Args:
            doc_path: Path to the input Word document
            
        Returns:
            Path to the generated PDF file
            
        Raises:
            RuntimeError: If conversion fails
            FileNotFoundError: If the output PDF is not created
        """
        # Ensure output directory exists
        self.out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.soffice,
            "--headless",
            "--invisible",
            "--convert-to", "pdf",
            "--outdir", str(self.out_dir),
            str(doc_path),
        ]
        
        # Capture output for debugging
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error("LibreOffice conversion failed: %s", result.stderr)
            raise RuntimeError(
                f"LibreOffice failed with return code {result.returncode}: {result.stderr}"
            )

        pdf_path = self.out_dir / f"{doc_path.stem}.pdf"
        if not pdf_path.exists():
            logger.error("Expected PDF not found: %s", pdf_path)
            logger.error("Output directory contents: %s", list(self.out_dir.iterdir()))
            raise FileNotFoundError(
                f"Conversion failed: Expected PDF not found at {pdf_path}"
            )
        
        logger.info("Successfully converted %s to %s", doc_path.name, pdf_path.name)
        return pdf_path