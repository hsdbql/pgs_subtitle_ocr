"""Base class for OCR engine implementations."""

from abc import ABC, abstractmethod
from PIL import Image


class OCREngine(ABC):
    """Abstract base class for OCR engines.
    
    All OCR engine implementations must inherit from this class and implement
    the required methods to ensure a consistent interface across different engines.
    """
    
    @abstractmethod
    def get_ocr_text(self, im: Image.Image) -> str:
        """Extract text from an image.
        
        Args:
            im: PIL Image object containing text to recognize
            
        Returns:
            Recognized text as a string. Multiple lines should be joined by newlines.
            Returns empty string if no text is detected or if processing fails.
        """
        pass
    
    @abstractmethod
    def quit(self) -> None:
        """Cleanup resources and release memory.
        
        This method should be called when the OCR engine is no longer needed
        to properly release any allocated resources (models, GPU memory, etc.).
        """
        pass
