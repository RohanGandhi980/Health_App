import re
from transformers import pipeline

hindi_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")

def multilingual_alert(message: str, target_lang: str) -> str:
    if target_lang == "English":
        return message
    elif target_lang == "Hindi":
        try:
            
            numbers = re.findall(r"[\d.]+", message)  
            protected = re.sub(r"[\d.]+", "<NUM>", message)

            translated = hindi_translator(protected)[0]['translation_text']

            for num in numbers:
                translated = translated.replace("<NUM>", num, 1)

            return translated
        except Exception as e:
            return f"[Hindi] translation unavailable: {message} (error: {e})"

    elif target_lang == "Assamese":
        return f"[Assamese translation unavailable] {message}"
