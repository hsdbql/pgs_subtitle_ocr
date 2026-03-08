import argparse
import textwrap
from pathlib import Path
from pgsocr.converters.supconvert import supconvert


def main():
    parser = argparse.ArgumentParser(
        prog="pgsocr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            OCR Engines:
            - paddleocr (default): Lightweight, supports multiple languages, good accuracy
            - florence2: AI-based, more accurate but resource-intensive, English only, requires GPU with large VRAM

            Examples:
            # Single file with PaddleOCR (default)
            pgsocr -i /path/to/file -o path/to/outputdir -m paddleocr -l eng jpn

            # Multiple files in a directory with Florence2
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
        help="Specify the OCR model to use.",
        choices=["paddleocr", "paddle", "florence2"],
        type=str.lower,
        default="paddleocr",
    )
    parser.add_argument(
        "-f",
        help="Specify the output format",
        choices=["srt", "ass"],
        type=str.lower,
        default="srt",
    )
    parser.add_argument(
        "-l",
        nargs="+",
        help="Specify the list of languages to use separated by spaces (e.g., eng, jpn, chi_sim).",
        type=str.lower,
        default=["eng"],
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
    if args.m in ["paddleocr", "paddle"]:
        from pgsocr.ocr.paddleocr_engine import PaddleOCREngine

        engine = PaddleOCREngine(langs)
    elif args.m == "florence2":
        from pgsocr.ocr.transformer_ocr_engines import Florence2OCREngine

        engine = Florence2OCREngine()
    else:
        raise ValueError(f"Unknown OCR engine '{args.m}' specified.")
    print("OCR engine loaded.")

    if inp.is_file():
        supconvert(str(inp), args.o, engine, args.f)
    elif inp.is_dir():
        for x in inp.iterdir():
            supconvert(str(x), args.o, engine, args.f)
    exit(0)
