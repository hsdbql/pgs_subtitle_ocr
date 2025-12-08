# pgsocr
Convert Blu-Ray SUP subtitles to SRT or ASS using AI Models or Tesseract.

### Prerequisites

If planning on using Tesseract, see: https://tesseract-ocr.github.io/tessdoc/Installation.html \
Make sure to install all the required language packs.

### Installation
    pip install pgsocr[ts]  # Tesseract
    # or
    pip install pgsocr[lm]  # Language models (florence2)
    # or
    pip install pgsocr[pp]  # PaddleOCR

### Usage:

    Options:
    -i: Specify the path to the SUP file or (batch mode) directory.
    -o: Specify the path to the output directory.
    -m: Specify the OCR engine to use (florence2 or tesseract or paddle).
    -l (Only if using Tesseract): Specify the list of languages to use separated by spaces. Defaults to English.
    -l (Only if using PaddleOCR): Specify a preffered language. Defaults to English/Chinese/Japanese auto-detect.
    -b (Only if using Tesseract): Specify a custom character blacklist for Tesseract. Enter an empty string to turn off the default blacklist.
    -f: Specify the output format (SRT or ASS). ASS output also has support for subtitle positioning.

    Note: The AI models are more accurate than Tesseract but far more resource heavy. A recent GPU with a large amount of VRAM is recommended.

    Examples:
    # Single file
    pgsocr -i /path/to/file -o path/to/outputdir -m tesseract -l eng jpn

    # Multiple files in a directory
    pgsocr -i /path/to/inputdir -o /path/to/outputdir -m florence2
