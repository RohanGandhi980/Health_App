# translate.py
from deep_translator import GoogleTranslator

SUPPORTED_LANGS = {
    "en": "english",
    "hi": "hindi",
    "as": "assamese",
    "bn": "bengali",
    "or": "odia",
    "ne": "nepali",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam"
}

def translate_text(text: str, to_lang: str, from_lang: str = "en") -> str:
    """Translate text using Google Translate (via deep-translator)."""
    try:
        if to_lang not in SUPPORTED_LANGS:
            return f"[Unsupported-Language:{to_lang}] {text}"
        if to_lang == from_lang:
            return text

        translated = GoogleTranslator(
            source=SUPPORTED_LANGS[from_lang],
            target=SUPPORTED_LANGS[to_lang]
        ).translate(text)
        return translated
    except Exception as e:
        print(f"[Translation Error] {e}")
        return f"[Fallback-English] {text}"

def multilingual_alert(text: str, target_lang: str = "en") -> str:
    """Wrapper to generate alerts in chosen language."""
    return translate_text(text, to_lang=target_lang, from_lang="en")


if __name__ == "__main__":
    sample = "Village 1: High outbreak risk. Please boil water."
    print("Hindi:", multilingual_alert(sample, "hi"))
    print("Assamese:", multilingual_alert(sample, "as"))
