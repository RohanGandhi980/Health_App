import re
from transformers import pipeline

# Load only English ↔ Hindi translator
hindi_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")

def multilingual_alert(message: str, target_lang: str) -> str:
    """
    Translates alerts into Hindi. English returns the original message.
    """
    if target_lang == "English":
        return message

    try:
        # Step 1: Protect numbers and punctuation
        placeholders = {}
        idx = 0

        def replacer(m):
            nonlocal idx
            key = f"<NUM{idx}>"
            placeholders[key] = m.group()
            idx += 1
            return key

        protected = re.sub(r"[\d.]+", replacer, message)

        # Step 2: Translate only to Hindi
        if target_lang == "Hindi":
            translated = hindi_translator(protected)[0]["translation_text"]
        else:
            return message  # fallback for unsupported langs

        # Step 3: Restore placeholders back
        for key, val in placeholders.items():
            translated = translated.replace(key, val)

        return translated

    except Exception as e:
        return f"[{target_lang}] translation unavailable: {message} (error: {e})"
