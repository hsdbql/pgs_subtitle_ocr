import argparse
import textwrap
from pathlib import Path
from pgsocr.supconvert import supconvert


def main():
    parser = argparse.ArgumentParser(
        prog="pgsocr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Note: The AI models are more accurate than Tesseract but far more resource heavy. A recent GPU with a large amount of VRAM is recommended.

            Examples:
            # Single file
            pgsocr -i /path/to/file -o path/to/outputdir -m tesseract -l eng jpn

            # Multiple files in a directory
            pgsocr -i /path/to/inputdir -o /path/to/outputdir -m florence2
        """
        ),
    )
    parser.add_argument(
        "-i",
        help="Specify the path to the SUP file or (batch mode) directory.",
        required=True,
    )
    parser.add_argument(
        "-o", help="Specify the path to the output directory.", required=True
    )
    parser.add_argument(
        "-m",
        help="Specify the OCR engine to use (florence2/tesseract/paddle).",
        choices=["tesseract", "florence2", "paddle"],
        type=str.lower,
        default="tesseract",
    )
    parser.add_argument(
        "-f",
        help="Specify the output format (SRT or ASS).",
        choices=["srt", "ass"],
        type=str.lower,
        default="srt",
    )
    parser.add_argument(
        "-l",
        nargs="+",
        help="(Only if using Tesseract or PaddleOCR) Specify preferred languages. Usually defaults to English.",
        type=str.lower,
        default=["eng"],
    )
    parser.add_argument(
        "-b",
        help="(Only if using Tesseract) Specify a custom character blacklist for Tesseract. Enter an empty string to turn off the default blacklist.",
        default="|`´®",
    )
    args = parser.parse_args()

    inp = Path(args.i)
    if not inp.exists():
        print("Input file not found, make sure you have specified the correct path.")
        exit(1)
    op = Path(args.o)
    if not op.exists() or not op.is_dir():
        print(
            "Output directory not found, make sure you have specified the correct path."
        )
        exit(1)

    langs = args.l

    print("Loading OCR engine...")
    if args.m == "tesseract":
        from .tesseract_ocr_engine import TesseractOCREngine

        engine = TesseractOCREngine(langs, args.b)
    elif args.m == "florence2":
        from .transformer_ocr_engines import Florence2OCREngine

        engine = Florence2OCREngine()
    elif args.m == "paddle":
        from .paddle_ocr_engine import PaddleOCREngine

        engine = PaddleOCREngine(langs, args.b)
    else:
        raise ValueError(f"Unknown OCR engine '{args.m}' specified.")
    print("OCR engine loaded.")

    if inp.is_file():
        supconvert(str(inp), args.o, engine, args.f)
    elif inp.is_dir():
        for x in inp.iterdir():
            supconvert(str(x), args.o, engine, args.f)
    exit(0)
