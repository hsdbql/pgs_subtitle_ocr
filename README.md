# pgsocr
Convert Blu-Ray SUP subtitles to SRT or ASS using AI Language Models or PaddleOCR.

### Installation
    pip install pgsocr
    or
    pip install pgsocr[lm] (for the AI language models)

PaddleOCR is included by default and will automatically download required language models on first use.

### Usage:

    Options:
    -i: Specify the path to the SUP file or (batch mode) directory.
    -o: Specify the path to the output directory.
    -m: Specify the OCR engine to use (paddleocr or florence2). Defaults to paddleocr.
    -l: (Only if using PaddleOCR) Specify the list of languages to use separated by spaces. Defaults to English.
    -f: Specify the output format (SRT or ASS). ASS output also has support for subtitle positioning.

### Supported Languages

PaddleOCR supports the following languages (use these codes with the -l flag):
- English: en, eng
- Chinese (Simplified): ch, chi_sim, zh
- Japanese: ja, jpn
- Korean: ko, kor
- French: fr
- German: de
- Spanish: es
- Portuguese: pt
- Russian: ru
- Italian: it

Additional languages may be supported. Language models are automatically downloaded on first use.

### OCR Engine Comparison

| Feature | PaddleOCR | Florence2 |
|---------|-----------|-----------|
| Accuracy | Good | Excellent |
| Speed | Fast (~500ms per image on CPU) | Slower |
| Resource Usage | Lightweight | Heavy (requires GPU with large VRAM) |
| Installation | Included by default | Requires [lm] extra |
| Language Support | Multiple languages with -l flag | Automatic language detection |

Note: Florence2 is more accurate than PaddleOCR but far more resource heavy. A recent GPU with a large amount of VRAM is recommended for Florence2.

### Examples:

    # Single file with PaddleOCR (default)
    pgsocr -i /path/to/file -o /path/to/outputdir -m paddleocr -l eng

    # Multiple languages with PaddleOCR
    pgsocr -i /path/to/file -o /path/to/outputdir -m paddleocr -l eng jpn

    # Multiple files in a directory with PaddleOCR
    pgsocr -i /path/to/inputdir -o /path/to/outputdir -m paddleocr

    # Using Florence2 AI model for higher accuracy
    pgsocr -i /path/to/inputdir -o /path/to/outputdir -m florence2
