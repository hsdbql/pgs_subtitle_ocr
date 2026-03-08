"""CLI integration tests for PaddleOCR engine selection and argument parsing."""

import pytest
import argparse
import sys
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
from io import StringIO


class TestCLIEngineSelection:
    """Test CLI accepts PaddleOCR engine names."""
    
    def test_cli_accepts_paddleocr_engine_name(self, tmp_path):
        """Test that CLI accepts '-m paddleocr' flag."""
        # Create temporary input file and output directory
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Mock PaddleOCR class and supconvert
        with patch('paddleocr.PaddleOCR') as mock_paddle, \
             patch('pgsocr.converters.supconvert.supconvert') as mock_supconvert:
            
            # Setup mock PaddleOCR instance
            mock_paddle_instance = MagicMock()
            mock_paddle.return_value = mock_paddle_instance
            
            # Simulate CLI arguments
            test_args = [
                'pgsocr',
                '-i', str(input_file),
                '-o', str(output_dir),
                '-m', 'paddleocr',
                '-l', 'eng'
            ]
            
            with patch.object(sys, 'argv', test_args):
                from pgsocr.cli.main import main
                
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                # Verify successful exit
                assert exc_info.value.code == 0
                
                # Verify PaddleOCR was initialized with correct language
                mock_paddle.assert_called_once()
                call_kwargs = mock_paddle.call_args[1]
                assert call_kwargs['lang'] == 'en'
    
    def test_cli_accepts_paddle_engine_alias(self, tmp_path):
        """Test that CLI accepts '-m paddle' as an alias."""
        # Create temporary input file and output directory
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Mock PaddleOCR class and supconvert
        with patch('paddleocr.PaddleOCR') as mock_paddle, \
             patch('pgsocr.converters.supconvert.supconvert') as mock_supconvert:
            
            # Setup mock PaddleOCR instance
            mock_paddle_instance = MagicMock()
            mock_paddle.return_value = mock_paddle_instance
            
            # Simulate CLI arguments with 'paddle' alias
            test_args = [
                'pgsocr',
                '-i', str(input_file),
                '-o', str(output_dir),
                '-m', 'paddle',
                '-l', 'eng'
            ]
            
            with patch.object(sys, 'argv', test_args):
                from pgsocr.cli.main import main
                
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                # Verify successful exit
                assert exc_info.value.code == 0
                
                # Verify PaddleOCR was initialized
                mock_paddle.assert_called_once()
    
    def test_cli_engine_name_case_insensitive(self, tmp_path):
        """Test that engine name is case-insensitive."""
        # Create temporary input file and output directory
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Mock PaddleOCR class and supconvert
        with patch('paddleocr.PaddleOCR') as mock_paddle, \
             patch('pgsocr.converters.supconvert.supconvert') as mock_supconvert:
            
            # Setup mock PaddleOCR instance
            mock_paddle_instance = MagicMock()
            mock_paddle.return_value = mock_paddle_instance
            
            # Test with uppercase
            test_args = [
                'pgsocr',
                '-i', str(input_file),
                '-o', str(output_dir),
                '-m', 'PADDLEOCR',
                '-l', 'eng'
            ]
            
            with patch.object(sys, 'argv', test_args):
                from pgsocr.cli.main import main
                
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                # Verify successful exit
                assert exc_info.value.code == 0
                
                # Verify PaddleOCR was initialized
                mock_paddle.assert_called_once()


class TestCLILanguageCodePropagation:
    """Test CLI passes language codes to the engine."""
    
    def test_cli_passes_single_language_code(self, tmp_path):
        """Test that CLI passes single language code to engine."""
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        with patch('paddleocr.PaddleOCR') as mock_paddle, \
             patch('pgsocr.converters.supconvert.supconvert') as mock_supconvert:
            
            mock_paddle_instance = MagicMock()
            mock_paddle.return_value = mock_paddle_instance
            
            test_args = [
                'pgsocr',
                '-i', str(input_file),
                '-o', str(output_dir),
                '-m', 'paddleocr',
                '-l', 'jpn'
            ]
            
            with patch.object(sys, 'argv', test_args):
                from pgsocr.cli.main import main
                
                with pytest.raises(SystemExit):
                    main()
                
                # Verify language code was passed to PaddleOCR
                mock_paddle.assert_called_once()
                call_kwargs = mock_paddle.call_args[1]
                assert call_kwargs['lang'] == 'japan'  # jpn maps to 'japan'
    
    def test_cli_passes_multiple_language_codes(self, tmp_path):
        """Test that CLI passes multiple language codes to engine."""
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        with patch('paddleocr.PaddleOCR') as mock_paddle, \
             patch('pgsocr.converters.supconvert.supconvert') as mock_supconvert:
            
            mock_paddle_instance = MagicMock()
            mock_paddle.return_value = mock_paddle_instance
            
            test_args = [
                'pgsocr',
                '-i', str(input_file),
                '-o', str(output_dir),
                '-m', 'paddleocr',
                '-l', 'eng', 'jpn', 'chi_sim'
            ]
            
            with patch.object(sys, 'argv', test_args):
                from pgsocr.cli.main import main
                
                with pytest.raises(SystemExit):
                    main()
                
                # Verify first language code was passed to PaddleOCR (it uses first language)
                mock_paddle.assert_called_once()
                call_kwargs = mock_paddle.call_args[1]
                assert call_kwargs['lang'] == 'en'  # First language: eng maps to 'en'
    
    def test_cli_defaults_to_eng_when_no_language_specified(self, tmp_path):
        """Test that CLI defaults to 'eng' when no language is specified."""
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        with patch('paddleocr.PaddleOCR') as mock_paddle, \
             patch('pgsocr.converters.supconvert.supconvert') as mock_supconvert:
            
            mock_paddle_instance = MagicMock()
            mock_paddle.return_value = mock_paddle_instance
            
            # No -l argument provided
            test_args = [
                'pgsocr',
                '-i', str(input_file),
                '-o', str(output_dir),
                '-m', 'paddleocr'
            ]
            
            with patch.object(sys, 'argv', test_args):
                from pgsocr.cli.main import main
                
                with pytest.raises(SystemExit):
                    main()
                
                # Verify default language 'eng' was passed (maps to 'en')
                mock_paddle.assert_called_once()
                call_kwargs = mock_paddle.call_args[1]
                assert call_kwargs['lang'] == 'en'
    
    def test_cli_language_codes_case_insensitive(self, tmp_path):
        """Test that language codes are converted to lowercase."""
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        with patch('paddleocr.PaddleOCR') as mock_paddle, \
             patch('pgsocr.converters.supconvert.supconvert') as mock_supconvert:
            
            mock_paddle_instance = MagicMock()
            mock_paddle.return_value = mock_paddle_instance
            
            test_args = [
                'pgsocr',
                '-i', str(input_file),
                '-o', str(output_dir),
                '-m', 'paddleocr',
                '-l', 'ENG', 'JPN'
            ]
            
            with patch.object(sys, 'argv', test_args):
                from pgsocr.cli.main import main
                
                with pytest.raises(SystemExit):
                    main()
                
                # Verify language codes were converted to lowercase and mapped
                mock_paddle.assert_called_once()
                call_kwargs = mock_paddle.call_args[1]
                assert call_kwargs['lang'] == 'en'  # ENG -> eng -> en


class TestCLIInvalidEngineRejection:
    """Test CLI rejects invalid engine names."""
    
    def test_cli_rejects_invalid_engine_name(self, tmp_path, capsys):
        """Test that CLI rejects invalid engine names with error message."""
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        test_args = [
            'pgsocr',
            '-i', str(input_file),
            '-o', str(output_dir),
            '-m', 'invalid_engine',
            '-l', 'eng'
        ]
        
        with patch.object(sys, 'argv', test_args):
            from pgsocr.cli.main import main
            
            # argparse will call sys.exit(2) for invalid choice
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            # Verify exit code indicates error
            assert exc_info.value.code == 2
            
            # Capture stderr to check error message
            captured = capsys.readouterr()
            error_output = captured.err
            
            # Verify error message mentions invalid choice
            assert 'invalid choice' in error_output.lower()
            assert 'invalid_engine' in error_output
    
    def test_cli_rejects_tesseract_engine_name(self, tmp_path, capsys):
        """Test that CLI rejects 'tesseract' as it's no longer supported."""
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        test_args = [
            'pgsocr',
            '-i', str(input_file),
            '-o', str(output_dir),
            '-m', 'tesseract',
            '-l', 'eng'
        ]
        
        with patch.object(sys, 'argv', test_args):
            from pgsocr.cli.main import main
            
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            # Verify exit code indicates error
            assert exc_info.value.code == 2
            
            # Capture stderr to check error message
            captured = capsys.readouterr()
            error_output = captured.err
            
            # Verify error message mentions invalid choice
            assert 'invalid choice' in error_output.lower()
            assert 'tesseract' in error_output
    
    def test_cli_error_lists_valid_engine_choices(self, tmp_path, capsys):
        """Test that error message lists valid engine choices."""
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        test_args = [
            'pgsocr',
            '-i', str(input_file),
            '-o', str(output_dir),
            '-m', 'unknown',
            '-l', 'eng'
        ]
        
        with patch.object(sys, 'argv', test_args):
            from pgsocr.cli.main import main
            
            with pytest.raises(SystemExit):
                main()
            
            captured = capsys.readouterr()
            error_output = captured.err
            
            # Verify valid choices are listed
            assert 'paddleocr' in error_output
            assert 'paddle' in error_output
            assert 'florence2' in error_output


class TestCLIHelpText:
    """Test CLI help text content."""
    
    def test_help_text_contains_paddleocr(self, capsys):
        """Test that help text contains 'paddleocr'."""
        test_args = ['pgsocr', '--help']
        
        with patch.object(sys, 'argv', test_args):
            from pgsocr.cli.main import main
            
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            # Help exits with code 0
            assert exc_info.value.code == 0
            
            captured = capsys.readouterr()
            help_output = captured.out
            
            # Verify 'paddleocr' appears in help text
            assert 'paddleocr' in help_output.lower()
    
    def test_help_text_does_not_contain_tesseract(self, capsys):
        """Test that help text does not contain 'tesseract'."""
        test_args = ['pgsocr', '--help']
        
        with patch.object(sys, 'argv', test_args):
            from pgsocr.cli.main import main
            
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            # Help exits with code 0
            assert exc_info.value.code == 0
            
            captured = capsys.readouterr()
            help_output = captured.out
            
            # Verify 'tesseract' does NOT appear in help text
            assert 'tesseract' not in help_output.lower()
    
    def test_help_text_describes_paddleocr_as_default(self, capsys):
        """Test that help text describes PaddleOCR as the default engine."""
        test_args = ['pgsocr', '--help']
        
        with patch.object(sys, 'argv', test_args):
            from pgsocr.cli.main import main
            
            with pytest.raises(SystemExit):
                main()
            
            captured = capsys.readouterr()
            help_output = captured.out
            
            # Verify PaddleOCR is mentioned as default
            assert 'paddleocr' in help_output.lower()
            assert 'default' in help_output.lower()
    
    def test_help_text_describes_paddle_alias(self, capsys):
        """Test that help text mentions 'paddle' as an option."""
        test_args = ['pgsocr', '--help']
        
        with patch.object(sys, 'argv', test_args):
            from pgsocr.cli.main import main
            
            with pytest.raises(SystemExit):
                main()
            
            captured = capsys.readouterr()
            help_output = captured.out
            
            # Verify 'paddle' appears as an option
            # It should be in the choices list for -m argument
            assert 'paddle' in help_output.lower()
    
    def test_help_text_describes_language_option(self, capsys):
        """Test that help text describes the -l language option."""
        test_args = ['pgsocr', '--help']
        
        with patch.object(sys, 'argv', test_args):
            from pgsocr.cli.main import main
            
            with pytest.raises(SystemExit):
                main()
            
            captured = capsys.readouterr()
            help_output = captured.out
            
            # Verify language option is documented
            assert '-l' in help_output
            assert 'language' in help_output.lower()
    
    def test_help_text_shows_usage_examples(self, capsys):
        """Test that help text includes usage examples."""
        test_args = ['pgsocr', '--help']
        
        with patch.object(sys, 'argv', test_args):
            from pgsocr.cli.main import main
            
            with pytest.raises(SystemExit):
                main()
            
            captured = capsys.readouterr()
            help_output = captured.out
            
            # Verify examples section exists
            assert 'example' in help_output.lower()
            
            # Verify example shows PaddleOCR usage
            assert 'pgsocr' in help_output.lower()
            assert '-i' in help_output
            assert '-o' in help_output
    
    def test_help_text_describes_ocr_engines(self, capsys):
        """Test that help text describes available OCR engines."""
        test_args = ['pgsocr', '--help']
        
        with patch.object(sys, 'argv', test_args):
            from pgsocr.cli.main import main
            
            with pytest.raises(SystemExit):
                main()
            
            captured = capsys.readouterr()
            help_output = captured.out
            
            # Verify OCR engines section exists
            assert 'ocr engine' in help_output.lower()
            
            # Verify both engines are described
            assert 'paddleocr' in help_output.lower()
            assert 'florence2' in help_output.lower()


class TestCLIDefaultBehavior:
    """Test CLI default behavior."""
    
    def test_cli_defaults_to_paddleocr_engine(self, tmp_path):
        """Test that CLI defaults to PaddleOCR when no engine specified."""
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        with patch('paddleocr.PaddleOCR') as mock_paddle, \
             patch('pgsocr.converters.supconvert.supconvert') as mock_supconvert:
            
            mock_paddle_instance = MagicMock()
            mock_paddle.return_value = mock_paddle_instance
            
            # No -m argument provided
            test_args = [
                'pgsocr',
                '-i', str(input_file),
                '-o', str(output_dir)
            ]
            
            with patch.object(sys, 'argv', test_args):
                from pgsocr.cli.main import main
                
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                # Verify successful exit
                assert exc_info.value.code == 0
                
                # Verify PaddleOCR was initialized (default engine)
                mock_paddle.assert_called_once()
    
    def test_cli_defaults_to_srt_format(self, tmp_path):
        """Test that CLI defaults to SRT output format."""
        input_file = tmp_path / "test.sup"
        input_file.write_bytes(b"dummy content")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        with patch('paddleocr.PaddleOCR') as mock_paddle, \
             patch('pgsocr.converters.supconvert.supconvert') as mock_supconvert:
            
            mock_paddle_instance = MagicMock()
            mock_paddle.return_value = mock_paddle_instance
            
            # No -f argument provided
            test_args = [
                'pgsocr',
                '-i', str(input_file),
                '-o', str(output_dir),
                '-m', 'paddleocr'
            ]
            
            with patch.object(sys, 'argv', test_args):
                from pgsocr.cli.main import main
                
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                # Verify successful exit (even if file is not valid SUP)
                assert exc_info.value.code == 0
