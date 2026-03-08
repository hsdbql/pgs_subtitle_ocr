"""Integration tests for supconvert.py error handling."""

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from PIL import Image
import tempfile
import os


class TestSupconvertErrorHandling:
    """Test error handling in supconvert batch processing."""
    
    def test_supconvert_continues_after_ocr_failure(self, tmp_path):
        """Test that supconvert continues processing after individual OCR failure."""
        # Import first to avoid import issues
        from pgsocr.converters.supconvert import supconvert
        
        # Create a mock OCR engine that fails on second image
        mock_engine = Mock()
        call_count = [0]
        
        def mock_get_ocr_text(img):
            call_count[0] += 1
            if call_count[0] == 2:
                # Fail on second image
                raise Exception("OCR processing failed")
            return f"Text {call_count[0]}"
        
        mock_engine.get_ocr_text = mock_get_ocr_text
        mock_engine.quit = Mock()
        
        # Create mock PGStream and image objects
        mock_img_obj1 = Mock()
        mock_img_obj1.img = Image.new('RGB', (100, 50), color='white')
        mock_img_obj1.start_ms = 1000
        mock_img_obj1.end_ms = 2000
        mock_img_obj1.x_pos = 100
        mock_img_obj1.y_pos = 200
        
        mock_img_obj2 = Mock()
        mock_img_obj2.img = Image.new('RGB', (100, 50), color='white')
        mock_img_obj2.start_ms = 3000
        mock_img_obj2.end_ms = 4000
        mock_img_obj2.x_pos = 100
        mock_img_obj2.y_pos = 200
        
        mock_img_obj3 = Mock()
        mock_img_obj3.img = Image.new('RGB', (100, 50), color='white')
        mock_img_obj3.start_ms = 5000
        mock_img_obj3.end_ms = 6000
        mock_img_obj3.x_pos = 100
        mock_img_obj3.y_pos = 200
        
        # Mock the extract_images generator
        def mock_extract_images(supfile):
            yield mock_img_obj1
            yield mock_img_obj2
            yield mock_img_obj3
        
        # Create temporary input and output paths
        input_path = tmp_path / "test.sup"
        input_path.write_bytes(b"dummy SUP content")
        output_path = tmp_path / "output"
        output_path.mkdir()
        
        # Mock PGStream
        with patch('pgsocr.converters.supconvert.PGStream') as mock_pgstream_class, \
             patch('pgsocr.utils.img_utils.extract_images', side_effect=mock_extract_images), \
             patch('pgsocr.utils.img_utils.preprocess_image', side_effect=lambda x: x), \
             patch('builtins.print') as mock_print:
            
            # Setup mock PGStream instance
            mock_pgstream = Mock()
            mock_pgstream.file_name = "test.sup"
            mock_pgstream.res_width = 1920
            mock_pgstream.res_height = 1080
            mock_pgstream_class.return_value = mock_pgstream
            
            supconvert(
                in_path=str(input_path),
                out_path=str(output_path),
                ocr_engine=mock_engine,
                fmt="srt"
            )
            
            # Verify OCR was called 3 times (all images processed)
            assert call_count[0] == 3
            
            # Verify quit was called
            mock_engine.quit.assert_called_once()
            
            # Verify warning was printed for the failed image
            print_calls = [str(call) for call in mock_print.call_args_list]
            warning_found = any('Warning' in str(call) and 'OCR failed' in str(call) for call in print_calls)
            assert warning_found, "Expected warning message for failed OCR"
            
            # Verify the warning mentions the timestamp
            warning_with_timestamp = any('3000ms' in str(call) for call in print_calls)
            assert warning_with_timestamp, "Expected warning to include timestamp"
            
            # Verify output file was created
            output_file = output_path / "test.srt"
            assert output_file.exists(), "Output file should be created"
            
            # Read and verify output content
            content = output_file.read_text()
            
            # Should have 3 subtitle entries (even though one failed)
            assert content.count('\n\n') >= 2, "Should have multiple subtitle entries"
            
            # First subtitle should have text
            assert "Text 1" in content, "First subtitle should have OCR text"
            
            # Third subtitle should have text
            assert "Text 3" in content, "Third subtitle should have OCR text"
            
            # Second subtitle should be present but with empty text (or just timecode)
            # The entry should still exist in the output
            assert "00:00:03,000 --> 00:00:04,000" in content, "Second subtitle timecode should exist"
    
    def test_supconvert_logs_error_details(self, tmp_path):
        """Test that supconvert logs detailed error information."""
        from pgsocr.converters.supconvert import supconvert
        
        # Create a mock OCR engine that raises a specific error
        mock_engine = Mock()
        mock_engine.get_ocr_text.side_effect = ValueError("Invalid image format")
        mock_engine.quit = Mock()
        
        # Create mock image object
        mock_img_obj = Mock()
        mock_img_obj.img = Image.new('RGB', (100, 50), color='white')
        mock_img_obj.start_ms = 1000
        mock_img_obj.end_ms = 2000
        mock_img_obj.x_pos = 100
        mock_img_obj.y_pos = 200
        
        def mock_extract_images(supfile):
            yield mock_img_obj
        
        input_path = tmp_path / "test.sup"
        input_path.write_bytes(b"dummy SUP content")
        output_path = tmp_path / "output"
        output_path.mkdir()
        
        with patch('pgsocr.converters.supconvert.PGStream') as mock_pgstream_class, \
             patch('pgsocr.utils.img_utils.extract_images', side_effect=mock_extract_images), \
             patch('pgsocr.utils.img_utils.preprocess_image', side_effect=lambda x: x), \
             patch('builtins.print') as mock_print:
            
            mock_pgstream = Mock()
            mock_pgstream.file_name = "test.sup"
            mock_pgstream.res_width = 1920
            mock_pgstream.res_height = 1080
            mock_pgstream_class.return_value = mock_pgstream
            
            supconvert(
                in_path=str(input_path),
                out_path=str(output_path),
                ocr_engine=mock_engine,
                fmt="srt"
            )
            
            # Verify error details were printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            error_message_found = any('Invalid image format' in str(call) for call in print_calls)
            assert error_message_found, "Expected error message to include exception details"
    
    def test_supconvert_preserves_pgs_parsing_warnings(self, tmp_path):
        """Test that supconvert preserves existing PGS parsing warnings."""
        # This test verifies that our error handling doesn't interfere with
        # existing warning behavior from PGS parsing
        from pgsocr.converters.supconvert import supconvert
        
        mock_engine = Mock()
        mock_engine.get_ocr_text.return_value = "Test text"
        mock_engine.quit = Mock()
        
        input_path = tmp_path / "test.sup"
        input_path.write_bytes(b"dummy SUP content")
        output_path = tmp_path / "output"
        output_path.mkdir()
        
        # Mock PGStream to raise a warning during initialization
        with patch('pgsocr.converters.supconvert.PGStream') as mock_pgstream_class, \
             patch('builtins.print') as mock_print:
            
            # Simulate PGStream raising a warning
            mock_pgstream = Mock()
            mock_pgstream.file_name = "test.sup"
            mock_pgstream.res_width = 1920
            mock_pgstream.res_height = 1080
            mock_pgstream_class.return_value = mock_pgstream
            
            # Mock extract_images to return empty (no images)
            with patch('pgsocr.utils.img_utils.extract_images', return_value=iter([])):
                supconvert(
                    in_path=str(input_path),
                    out_path=str(output_path),
                    ocr_engine=mock_engine,
                    fmt="srt"
                )
                
                # Verify supconvert completed successfully
                mock_engine.quit.assert_called_once()
                
                # Output file should be created
                output_file = output_path / "test.srt"
                assert output_file.exists()
    
    def test_supconvert_handles_preprocessing_errors(self, tmp_path):
        """Test that supconvert handles errors during image preprocessing."""
        from pgsocr.converters.supconvert import supconvert
        
        mock_engine = Mock()
        mock_engine.get_ocr_text.return_value = "Test text"
        mock_engine.quit = Mock()
        
        mock_img_obj = Mock()
        mock_img_obj.img = Image.new('RGB', (100, 50), color='white')
        mock_img_obj.start_ms = 1000
        mock_img_obj.end_ms = 2000
        mock_img_obj.x_pos = 100
        mock_img_obj.y_pos = 200
        
        def mock_extract_images(supfile):
            yield mock_img_obj
        
        input_path = tmp_path / "test.sup"
        input_path.write_bytes(b"dummy SUP content")
        output_path = tmp_path / "output"
        output_path.mkdir()
        
        with patch('pgsocr.converters.supconvert.PGStream') as mock_pgstream_class, \
             patch('pgsocr.utils.img_utils.extract_images', side_effect=mock_extract_images), \
             patch('pgsocr.utils.img_utils.preprocess_image', side_effect=Exception("Preprocessing failed")), \
             patch('builtins.print') as mock_print:
            
            mock_pgstream = Mock()
            mock_pgstream.file_name = "test.sup"
            mock_pgstream.res_width = 1920
            mock_pgstream.res_height = 1080
            mock_pgstream_class.return_value = mock_pgstream
            
            supconvert(
                in_path=str(input_path),
                out_path=str(output_path),
                ocr_engine=mock_engine,
                fmt="srt"
            )
            
            # Verify warning was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            warning_found = any('Warning' in str(call) and 'OCR failed' in str(call) for call in print_calls)
            assert warning_found
            
            # Verify processing completed
            mock_engine.quit.assert_called_once()
    
    def test_supconvert_ass_format_error_handling(self, tmp_path):
        """Test error handling works with ASS format output."""
        from pgsocr.converters.supconvert import supconvert
        
        mock_engine = Mock()
        call_count = [0]
        
        def mock_get_ocr_text(img):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("OCR failed")
            return f"Text {call_count[0]}"
        
        mock_engine.get_ocr_text = mock_get_ocr_text
        mock_engine.quit = Mock()
        
        # Create mock image objects
        mock_img_obj1 = Mock()
        mock_img_obj1.img = Image.new('RGB', (100, 50), color='white')
        mock_img_obj1.start_ms = 1000
        mock_img_obj1.end_ms = 2000
        mock_img_obj1.x_pos = 100
        mock_img_obj1.y_pos = 200
        
        mock_img_obj2 = Mock()
        mock_img_obj2.img = Image.new('RGB', (100, 50), color='white')
        mock_img_obj2.start_ms = 3000
        mock_img_obj2.end_ms = 4000
        mock_img_obj2.x_pos = 100
        mock_img_obj2.y_pos = 200
        
        def mock_extract_images(supfile):
            yield mock_img_obj1
            yield mock_img_obj2
        
        input_path = tmp_path / "test.sup"
        input_path.write_bytes(b"dummy SUP content")
        output_path = tmp_path / "output"
        output_path.mkdir()
        
        with patch('pgsocr.converters.supconvert.PGStream') as mock_pgstream_class, \
             patch('pgsocr.utils.img_utils.extract_images', side_effect=mock_extract_images), \
             patch('pgsocr.utils.img_utils.preprocess_image', side_effect=lambda x: x), \
             patch('builtins.print') as mock_print:
            
            mock_pgstream = Mock()
            mock_pgstream.file_name = "test.sup"
            mock_pgstream.res_width = 1920
            mock_pgstream.res_height = 1080
            mock_pgstream_class.return_value = mock_pgstream
            
            supconvert(
                in_path=str(input_path),
                out_path=str(output_path),
                ocr_engine=mock_engine,
                fmt="ass"
            )
            
            # Verify both images were processed
            assert call_count[0] == 2
            
            # Verify warning was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            warning_found = any('Warning' in str(call) for call in print_calls)
            assert warning_found
            
            # Verify output file was created
            output_file = output_path / "test.ass"
            assert output_file.exists()
            
            # Verify ASS format structure
            content = output_file.read_text()
            assert "[Script Info]" in content
            assert "[Events]" in content
            assert "Dialogue:" in content
