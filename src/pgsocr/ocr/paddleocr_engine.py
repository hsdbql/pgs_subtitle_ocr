"""PaddleOCR engine implementation for OCR processing."""

from PIL import Image
import numpy as np
from pgsocr.ocr.base import OCREngine

# Language code mapping from user-provided codes to PaddleOCR format
LANGUAGE_MAP = {
    # English
    'en': 'en',
    'eng': 'en',
    
    # Chinese
    'ch': 'ch',
    'chi_sim': 'ch',
    'zh': 'ch',
    'chinese': 'ch',
    
    # Japanese
    'ja': 'japan',
    'jpn': 'japan',
    'japanese': 'japan',
    
    # Korean
    'ko': 'korean',
    'kor': 'korean',
    'korean': 'korean',
    
    # French
    'fr': 'french',
    'fra': 'french',
    'french': 'french',
    
    # German
    'de': 'german',
    'deu': 'german',
    'ger': 'german',
    'german': 'german',
    
    # Spanish
    'es': 'spanish',
    'spa': 'spanish',
    'spanish': 'spanish',
    
    # Portuguese
    'pt': 'portuguese',
    'por': 'portuguese',
    'portuguese': 'portuguese',
    
    # Russian
    'ru': 'russian',
    'rus': 'russian',
    'russian': 'russian',
    
    # Italian
    'it': 'italian',
    'ita': 'italian',
    'italian': 'italian',
    
    # Arabic
    'ar': 'arabic',
    'ara': 'arabic',
    'arabic': 'arabic',
    
    # Hindi
    'hi': 'hindi',
    'hin': 'hindi',
    'hindi': 'hindi',
}


class PaddleOCREngine(OCREngine):
    """OCR engine implementation using PaddleOCR."""
    
    def __init__(self, languages: list[str]) -> None:
        """
        Initialize PaddleOCR engine with specified languages.
        
        Args:
            languages: List of ISO language codes (e.g., ['en', 'ch', 'ja'])
        
        Raises:
            ImportError: If paddleocr is not installed
            ValueError: If unsupported language code is provided
        """
        # Import PaddleOCR with helpful error message if not installed
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            print("PaddleOCR is not installed. Install it with:")
            print("  pip install paddleocr paddlepaddle")
            exit(1)
        
        # Validate and map language codes
        if not languages:
            languages = ['en']  # Default to English if no language specified
        
        # Use the first language for PaddleOCR (it supports one language at a time)
        primary_lang = languages[0].lower()
        
        if primary_lang not in LANGUAGE_MAP:
            supported = ', '.join(sorted(set(LANGUAGE_MAP.keys())))
            raise ValueError(
                f"Unsupported language code '{primary_lang}'. "
                f"Supported languages: {supported}"
            )
        
        paddle_lang = LANGUAGE_MAP[primary_lang]
        
        # Initialize PaddleOCR with configuration
        # Note: PaddleOCR 2.7+ automatically detects GPU availability
        try:
            self.paddle_ocr = PaddleOCR(
                lang=paddle_lang,
                use_angle_cls=True,  # Handle rotated text
            )
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {e}")
            print("This may be due to:")
            print("  - Missing language models (will auto-download on first use)")
            print("  - GPU configuration issues (try CPU mode)")
            print("  - Insufficient disk space for model downloads")
            exit(1)
    
    def get_ocr_text(self, im: Image.Image) -> str:
        """
        Extract text from an image using PaddleOCR.
        
        Args:
            im: PIL Image object containing text to recognize
            
        Returns:
            Recognized text as a string, with multiple lines joined by newlines.
            Returns empty string if no text is detected or if processing fails.
        """
        try:
            # Convert PIL Image to numpy array (RGB format)
            img_array = np.array(im)
            
            # Perform OCR
            result = self.paddle_ocr.ocr(img_array)
            
            if not result:
                return ""
            
            # Extract recognized texts from OCRResult
            # result is a list containing one OCRResult object per page
            ocr_result = result[0]
            
            # rec_texts is a list of strings, one per detected text region
            texts = ocr_result.get('rec_texts', [])
            if texts:
                return "\n".join(str(text) for text in texts if text).strip()
            
            return ""
        
        except Exception as e:
            print(f"Warning: Failed to process image with PaddleOCR: {e}")
            return ""
    
    def quit(self) -> None:
        """
        Cleanup resources. Currently a no-op for PaddleOCR compatibility.
        """
        pass
