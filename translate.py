from transformers import pipeline

# Only load English ↔ Hindi translator
hindi_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")

def multilingual_alert(message: str, target_lang: str) -> str:
    """
    Translates alerts into Hindi. English returns the original message.
    """
    if target_lang == "English":
        return message

    try:
        if target_lang == "Hindi":
            translated = hindi_translator(message)[0]["translation_text"]
            return translated
        else:
            return message  # fallback for unsupported languages

    except Exception as e:
        return f"[{target_lang}] translation unavailable: {message} (error: {e})"
