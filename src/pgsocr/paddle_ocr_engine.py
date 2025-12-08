import os
import tempfile
from PIL import Image
from paddleocr import PaddleOCR

# Mapping from Tesseract 3-letter codes (or common names) to PaddleOCR language codes
# Only storing differences. If not here, we pass the original code.
LANG_MAPPING = {
    # Likely to be default, and unnecessary to specify
    "eng": None,
    # Most Common
    "chi_sim": "ch",
    "zh": "ch",
    "fre": "fr",
    "ger": "de",
    "jpn": "japan",
    "ja": "japan",
    "kor": "korean",
    "ko": "korean",
    "chi_tra": "chinese_cht",
    "zh-tw": "chinese_cht",
    "zh-hk": "chinese_cht",
    # European
    "ita": "it",
    "spa": "es",
    "por": "pt",
    "rus": "ru",
    "ara": "ar",
    "hin": "hi",
    "hun": "hu",
    "srp_latn": "rs_latin",
    "ind": "id",
    "oci": "oc",
    "isl": "is",
    "lit": "lt",
    "mri": "mi",
    "afr": "af",
    "msa": "ms",
    "nld": "nl",
    "dut": "nl",
    "nor": "no",
    "pol": "pl",
    "slk": "sk",
    "slv": "sl",
    "ces": "cs",
    "cze": "cs",
    "cym": "cy",
    "sqi": "sq",
    "alb": "sq",
    "dan": "da",
    "swe": "sv",
    "est": "et",
    "swa": "sw",
    "gle": "ga",
    "tgl": "tl",
    "hrv": "hr",
    "tur": "tr",
    "uzb": "uz",
    "lat": "la",
    "bel": "be",
    "ukr": "uk",
    "tha": "th",
    "ell": "el",
    "aze": "az",
    "kur": "ku",
    "lav": "lv",
    "mlt": "mt",
    "pli": "pi",
    "ron": "ro",
    "rum": "ro",
    "vie": "vi",
    "fin": "fi",
    "eus": "eu",
    "baq": "eu",
    "glg": "gl",
    "ltz": "lb",
    "roh": "rm",
    "cat": "ca",
    "que": "qu",
    "tel": "te",
    "srp": "sr",
    "bul": "bg",
    "mon": "mn",
    "abk": "ab",
    "ava": "av",
    "che": "ce",
    "kaz": "kk",
    "kir": "ky",
    "tgk": "tg",
    "mkd": "mk",
    "mac": "mk",
    "tat": "tt",
    "chv": "cv",
    "bak": "ba",
    "kom": "kv",
    "oss": "os",
    "fas": "fa",
    "per": "fa",
    "uig": "ug",
    "urd": "ur",
    "pus": "ps",
    "snd": "sd",
    "mar": "mr",
    "nep": "ne",
    "bih": "bh",
    "san": "sa",
    "tam": "ta",
}


class PaddleOCREngine:
    def __init__(self, langs, blacklist=None):
        # Determine language
        target_lang = None
        if langs and len(langs) > 0:
            first_lang = langs[0].lower()
            # Use mapping if exists, otherwise use the original code
            target_lang = LANG_MAPPING.get(first_lang, first_lang)

        # Initialize PaddleOCR with parameters from the official demo / new API
        kwargs = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if target_lang:
            kwargs["lang"] = target_lang

        self.ocr = PaddleOCR(**kwargs)
        # Check for GPU availability if possible or let Paddle decide defaults

    def get_ocr_text(self, im: Image.Image):
        # The new PaddleOCR pipeline API (v3+) via predict() often requires a file path.
        # It's safest to save to a temp file as shown in the demo usage.

        # Create a temp file
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

        try:
            im.save(temp_path)

            # Predict
            # result is a list of results (usually one per image)
            results = self.ocr.predict(temp_path)

            texts = []
            if results:
                for res in results:
                    # Parse structure based on: {'res': {'rec_texts': [...], ...}}

                    content = None
                    # 1. Try attribute access (e.g. res.res)
                    if hasattr(res, "res"):
                        content = res.res
                    # 2. Try dict access (e.g. res['res'])
                    elif isinstance(res, dict) and "res" in res:
                        content = res["res"]
                    # 3. Fallback: maybe res is the content itself
                    else:
                        content = res

                    if isinstance(content, dict):
                        rec_texts = content.get("rec_texts", [])
                        if rec_texts:
                            texts.extend(rec_texts)

                    # Backwards compatibility check (direct attribute on res)
                    elif hasattr(res, "rec_texts") and res.rec_texts:
                        texts.extend(res.rec_texts)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return "\n".join(texts)

    def quit(self):
        # No explicit cleanup needed for PaddleOCR object
        pass
