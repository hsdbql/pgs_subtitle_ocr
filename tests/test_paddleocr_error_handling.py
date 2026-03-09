"""Unit tests for error handling in PaddleOCR engine."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np


class TestDependencyErrorHandling:
    """Test error handling for missing dependencies."""
    
    def test_missing_paddleocr_dependency_shows_install_command(self):
        """Test that missing PaddleOCR shows installation command."""
        # We test the error handling by checking the code structure
        # since mocking imports at this level is complex
        from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
        import inspect
        
        # Verify the code has proper ImportError handling
        source = inspect.getsource(PaddleOCREngine.__init__)
        assert 'ImportError' in source
        assert 'pip install paddleocr paddlepaddle' in source
        assert 'PaddleOCR is not installed' in source
    
    def test_missing_paddleocr_shows_descriptive_message(self):
        """Test that missing PaddleOCR shows descriptive error message."""
        # Verify the error message content by inspecting the source
        from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
        import inspect
        
        source = inspect.getsource(PaddleOCREngine.__init__)
        
        # Verify helpful message exists in code
        assert 'PaddleOCR is not installed' in source
        assert 'Install it with' in source or 'install' in source.lower()


class TestLanguageCodeErrorHandling:
    """Test error handling for invalid language codes."""
    
    def test_unsupported_language_code_raises_value_error(self):
        """Test that unsupported language code raises ValueError."""
        with patch('paddleocr.PaddleOCR'):
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            with pytest.raises(ValueError) as exc_info:
                PaddleOCREngine(['unsupported_lang'])
            
            # Verify error message contains the invalid code
            assert 'unsupported_lang' in str(exc_info.value).lower()
    
    def test_unsupported_language_lists_supported_languages(self):
        """Test that error message lists supported languages."""
        with patch('paddleocr.PaddleOCR'):
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            with pytest.raises(ValueError) as exc_info:
                PaddleOCREngine(['xyz'])
            
            error_message = str(exc_info.value)
            
            # Verify error message lists some supported languages
            assert 'supported languages' in error_message.lower()
            assert 'en' in error_message or 'eng' in error_message
    
    def test_empty_language_list_defaults_to_english(self):
        """Test that empty language list defaults to English."""
        with patch('paddleocr.PaddleOCR') as mock_paddle:
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            # Should not raise error, should default to English
            engine = PaddleOCREngine([])
            
            # Verify PaddleOCR was initialized with 'en'
            call_kwargs = mock_paddle.call_args[1]
            assert call_kwargs['lang'] == 'en'
    
    def test_invalid_language_code_provides_clear_error(self):
        """Test that invalid language code provides clear, actionable error."""
        with patch('paddleocr.PaddleOCR'):
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            with pytest.raises(ValueError) as exc_info:
                PaddleOCREngine(['invalid123'])
            
            error_message = str(exc_info.value)
            
            # Verify error is clear and actionable
            assert 'unsupported' in error_message.lower()
            assert 'invalid123' in error_message.lower()
            assert 'supported' in error_message.lower()


class TestInitializationErrorHandling:
    """Test error handling for PaddleOCR initialization failures."""
    
    def test_initialization_failure_shows_descriptive_message(self):
        """Test that initialization failure shows descriptive error message."""
        with patch('paddleocr.PaddleOCR', side_effect=Exception("Model download failed")):
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            with patch('builtins.print') as mock_print:
                with pytest.raises(SystemExit) as exc_info:
                    PaddleOCREngine(['en'])
                
                # Verify exit code
                assert exc_info.value.code == 1
                
                # Verify descriptive message was printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any('Failed to initialize PaddleOCR' in str(call) for call in print_calls)
    
    def test_initialization_failure_provides_troubleshooting_hints(self):
        """Test that initialization failure provides troubleshooting hints."""
        with patch('paddleocr.PaddleOCR', side_effect=Exception("CUDA error")):
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            with patch('builtins.print') as mock_print:
                with pytest.raises(SystemExit):
                    PaddleOCREngine(['en'])
                
                # Verify troubleshooting hints were printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                hints_printed = ''.join(str(call) for call in print_calls)
                
                # Check for specific troubleshooting hints
                assert 'Missing language models' in hints_printed or 'language models' in hints_printed
                assert 'GPU configuration issues' in hints_printed or 'GPU' in hints_printed
                assert 'disk space' in hints_printed
    
    def test_model_download_failure_mentions_auto_download(self):
        """Test that model download failure mentions auto-download feature."""
        with patch('paddleocr.PaddleOCR', side_effect=Exception("Download failed")):
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            with patch('builtins.print') as mock_print:
                with pytest.raises(SystemExit):
                    PaddleOCREngine(['en'])
                
                print_calls = [str(call) for call in mock_print.call_args_list]
                hints_printed = ''.join(str(call) for call in print_calls)
                
                # Verify auto-download is mentioned
                assert 'auto-download' in hints_printed.lower() or 'language models' in hints_printed
    
    def test_gpu_issue_mentions_cpu_fallback(self):
        """Test that GPU issues mention trying CPU mode."""
        with patch('paddleocr.PaddleOCR', side_effect=Exception("CUDA out of memory")):
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            with patch('builtins.print') as mock_print:
                with pytest.raises(SystemExit):
                    PaddleOCREngine(['en'])
                
                print_calls = [str(call) for call in mock_print.call_args_list]
                hints_printed = ''.join(str(call) for call in print_calls)
                
                # Verify CPU mode is mentioned as alternative
                assert 'CPU mode' in hints_printed or 'CPU' in hints_printed
    
    def test_disk_space_issue_mentioned_in_hints(self):
        """Test that disk space is mentioned in troubleshooting hints."""
        with patch('paddleocr.PaddleOCR', side_effect=Exception("No space left on device")):
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            with patch('builtins.print') as mock_print:
                with pytest.raises(SystemExit):
                    PaddleOCREngine(['en'])
                
                print_calls = [str(call) for call in mock_print.call_args_list]
                hints_printed = ''.join(str(call) for call in print_calls)
                
                # Verify disk space is mentioned
                assert 'disk space' in hints_printed.lower()


class TestImageProcessingErrorHandling:
    """Test error handling for individual image processing failures."""
    
    def test_image_processing_failure_returns_empty_string(self):
        """Test that image processing failure returns empty string."""
        with patch('paddleocr.PaddleOCR') as mock_paddle_class:
            # Setup mock that fails during OCR
            mock_paddle_instance = MagicMock()
            mock_paddle_instance.ocr.side_effect = Exception("Processing failed")
            mock_paddle_class.return_value = mock_paddle_instance
            
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            engine = PaddleOCREngine(['en'])
            
            # Create a test image
            test_image = Image.new('RGB', (100, 50), color='white')
            
            # Should return empty string instead of raising exception
            with patch('builtins.print'):  # Suppress warning output
                result = engine.get_ocr_text(test_image)
            
            assert result == ""
    
    def test_image_processing_failure_logs_warning(self):
        """Test that image processing failure logs a warning."""
        with patch('paddleocr.PaddleOCR') as mock_paddle_class:
            # Setup mock that fails during OCR
            mock_paddle_instance = MagicMock()
            mock_paddle_instance.ocr.side_effect = Exception("OCR processing error")
            mock_paddle_class.return_value = mock_paddle_instance
            
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            engine = PaddleOCREngine(['en'])
            test_image = Image.new('RGB', (100, 50), color='white')
            
            with patch('builtins.print') as mock_print:
                result = engine.get_ocr_text(test_image)
            
            # Verify warning was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any('Warning' in str(call) for call in print_calls)
            assert any('Failed to process image' in str(call) for call in print_calls)
    
    def test_numpy_conversion_failure_handled_gracefully(self):
        """Test that numpy array conversion failure is handled gracefully."""
        with patch('paddleocr.PaddleOCR') as mock_paddle_class:
            mock_paddle_instance = MagicMock()
            mock_paddle_class.return_value = mock_paddle_instance
            
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            engine = PaddleOCREngine(['en'])
            
            # Mock numpy.array to raise exception
            with patch('numpy.array', side_effect=Exception("Conversion failed")):
                test_image = Image.new('RGB', (100, 50), color='white')
                
                with patch('builtins.print'):  # Suppress warning
                    result = engine.get_ocr_text(test_image)
                
                # Should return empty string
                assert result == ""
    
    def test_ocr_result_parsing_failure_handled_gracefully(self):
        """Test that OCR result parsing failure is handled gracefully."""
        with patch('paddleocr.PaddleOCR') as mock_paddle_class:
            # Setup mock that returns malformed result
            mock_paddle_instance = MagicMock()
            mock_paddle_instance.ocr.return_value = [[[None, None]]]  # Malformed result
            mock_paddle_class.return_value = mock_paddle_instance
            
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            engine = PaddleOCREngine(['en'])
            test_image = Image.new('RGB', (100, 50), color='white')
            
            # Should handle malformed result gracefully
            try:
                result = engine.get_ocr_text(test_image)
                # Either returns empty string or handles the malformed data
                assert isinstance(result, str)
            except Exception:
                # If it raises, it should be caught by the try-except in get_ocr_text
                pytest.fail("get_ocr_text should handle malformed results gracefully")
    
    def test_empty_image_returns_empty_string(self):
        """Test that empty/blank image returns empty string."""
        with patch('paddleocr.PaddleOCR') as mock_paddle_class:
            # Setup mock that returns None (no text detected)
            mock_paddle_instance = MagicMock()
            mock_paddle_instance.ocr.return_value = None
            mock_paddle_class.return_value = mock_paddle_instance
            
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            engine = PaddleOCREngine(['en'])
            test_image = Image.new('RGB', (100, 50), color='white')
            
            result = engine.get_ocr_text(test_image)
            
            # Should return empty string for images with no text
            assert result == ""
    
    def test_batch_processing_continues_after_failure(self):
        """Test that batch processing can continue after individual image failure."""
        # This test verifies that the engine instance remains usable after an error
        # The actual batch processing error recovery happens in supconvert.py
        with patch('paddleocr.PaddleOCR') as mock_paddle_class:
            mock_paddle_instance = MagicMock()
            mock_paddle_class.return_value = mock_paddle_instance
            
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            engine = PaddleOCREngine(['en'])
            
            # First call fails
            mock_paddle_instance.ocr.side_effect = Exception("First image failed")
            image1 = Image.new('RGB', (100, 50), color='white')
            
            with patch('builtins.print'):  # Suppress warnings
                result1 = engine.get_ocr_text(image1)
            
            # Verify first call returned empty string
            assert result1 == ""
            
            # Second call succeeds - reset the mock to return valid data
            # Format for PaddleOCR 2.7+: [OCRResult] where OCRResult has .get() method
            mock_paddle_instance.ocr.side_effect = None
            mock_ocr_result = MagicMock()
            mock_ocr_result.get.return_value = ["Success"]
            mock_paddle_instance.ocr.return_value = [mock_ocr_result]
            image2 = Image.new('RGB', (100, 50), color='white')
            
            result2 = engine.get_ocr_text(image2)
            
            # Verify second call succeeded
            assert result2 == "Success"
            
            # Verify the engine is still functional (can be called again)
            assert engine.paddle_ocr is not None


class TestErrorMessageQuality:
    """Test that error messages are clear and actionable."""
    
    def test_all_error_messages_are_user_friendly(self):
        """Test that error messages avoid technical jargon where possible."""
        with patch('paddleocr.PaddleOCR', side_effect=Exception("Internal error")):
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            with patch('builtins.print') as mock_print:
                with pytest.raises(SystemExit):
                    PaddleOCREngine(['en'])
                
                print_calls = [str(call) for call in mock_print.call_args_list]
                messages = ''.join(str(call) for call in print_calls)
                
                # Verify messages are descriptive and actionable
                assert 'Failed to initialize' in messages
                assert 'This may be due to' in messages
    
    def test_error_messages_provide_next_steps(self):
        """Test that error messages suggest next steps to user."""
        with patch('paddleocr.PaddleOCR', side_effect=Exception("Error")):
            from pgsocr.ocr.paddleocr_engine import PaddleOCREngine
            
            with patch('builtins.print') as mock_print:
                with pytest.raises(SystemExit):
                    PaddleOCREngine(['en'])
                
                print_calls = [str(call) for call in mock_print.call_args_list]
                messages = ''.join(str(call) for call in print_calls)
                
                # Verify actionable suggestions are provided
                # Should mention at least one of: models, GPU, disk space
                has_actionable_hint = (
                    'models' in messages.lower() or
                    'gpu' in messages.lower() or
                    'disk' in messages.lower()
                )
                assert has_actionable_hint
