"""PaddleOCR engine implementation for OCR processing."""

from PIL import Image
import numpy as np
from pgsocr.ocr.base import OCREngine

# Language code mapping from non-standard/alternative codes to standard PaddleOCR format
LANGUAGE_MAP = {
    'ara': 'ar',
    'arabic': 'ar',
    'chi_sim': 'ch',
    'chinese': 'ch',
    'chinesesim': 'ch',
    'chinesetraditional': 'chinese_cht',
    'chs': 'ch',
    'cht': 'chinese_cht',
    'deu': 'de',
    'eng': 'en',
    'english': 'en',
    'fra': 'fr',
    'french': 'fr',
    'ger': 'de',
    'german': 'de',
    'hin': 'hi',
    'hindi': 'hi',
    'indonesian': 'id',
    'ita': 'it',
    'italian': 'it',
    'ja': 'japan',
    'japanese': 'japan',
    'jpn': 'japan',
    'ko': 'korean',
    'kor': 'korean',
    'malay': 'ms',
    'por': 'pt',
    'portuguese': 'pt',
    'rus': 'ru',
    'russian': 'ru',
    'spa': 'es',
    'spanish': 'es',
    'thai': 'th',
    'turkish': 'tr',
    'vietnamese': 'vi',
    'zh': 'ch',
}

# Standard PaddleOCR language codes supported by this engine
SUPPORTED_LANGUAGES = {
    'ab', 'ady', 'af', 'ang', 'ar', 'az', 'ba', 'bal', 'be', 'bg', 'bgc', 'bh',
    'bho', 'bs', 'bua', 'ca', 'ce', 'ch', 'chinese_cht', 'cs', 'cv', 'cy', 'da',
    'dar', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'ga', 'gom',
    'gl', 'hi', 'hr', 'hu', 'id', 'inh', 'is', 'it', 'japan', 'kaa', 'kbd',
    'kk', 'ko', 'korean', 'kv', 'ky', 'la', 'lb', 'lez', 'lki', 'lt', 'lv',
    'mah', 'mai', 'mhr', 'mi', 'mk', 'mn', 'mo', 'mr', 'ms', 'mt', 'ne', 'new',
    'nl', 'no', 'oc', 'os', 'pi', 'pl', 'ps', 'pt', 'qu', 'rm', 'ro', 'rs_cyrillic',
    'rs_latin', 'ru', 'sa', 'sah', 'sck', 'sd', 'sk', 'sl', 'sq', 'sv', 'sw',
    'ta', 'tab', 'te', 'tg', 'th', 'tl', 'tr', 'tt', 'tyv', 'udm', 'ug', 'uk',
    'ur', 'uz', 'vi', 'xal'
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
            languages = ['en']
        
        primary_lang = languages[0].lower()
        
        if primary_lang == 'auto':
            paddle_lang = None
        else:
            # Map alias to standard, or use as-is if already standard
            paddle_lang = LANGUAGE_MAP.get(primary_lang, primary_lang)
            
            if paddle_lang not in SUPPORTED_LANGUAGES:
                # Combine standard and alias keys for error message
                all_valid = sorted(list(SUPPORTED_LANGUAGES) + list(LANGUAGE_MAP.keys()))
                supported = ', '.join(all_valid)
                raise ValueError(
                    f"Unsupported language code '{primary_lang}'. "
                    f"Supported languages: {supported}"
                )
        
        # Initialize PaddleOCR with configuration
        # Note: PaddleOCR 2.7+ automatically detects GPU availability
        try:
            self.paddle_ocr = PaddleOCR(
                lang=paddle_lang,
                use_angle_cls=False,  # No Handle rotated text
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {e}")
            print("This may be due to:")
            print("  - Missing language models (will auto-download on first use)")
            print("  - GPU configuration issues (try CPU mode)")
            print("  - Insufficient disk space for model downloads")
            print("  - Conflicts with other cudnn libraries (try in a clean virtual environment or pip uninstall nvidia-cudnn-cu12)")
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
