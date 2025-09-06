from transformers import pipeline
# Hindi translator
hindi_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")


LANG_MODELS = {
    "Hindi": "Helsinki-NLP/opus-mt-en-hi"
    # Assamese removed because no model exists
}

def multilingual_alert(message: str, target_lang: str) -> str:
    if target_lang == "English":
        return message
    elif target_lang == "Hindi":
        try:
            translated = hindi_translator(message)
            return translated[0]['translation_text']
        except Exception as e:
            return f"[Hindi] translation unavailable: {message} (error: {e})"
    elif target_lang == "Assamese":
        # No direct Hugging Face model available
        return f"[Assamese translation unavailable] {message}"
