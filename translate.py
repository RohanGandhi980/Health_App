import re
from transformers import pipeline

# Hindi translator
hindi_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")

def multilingual_alert(message: str, target_lang: str) -> str:
    if target_lang == "English":
        return message

    elif target_lang == "Hindi":
        try:
            # Step 1: Find all numbers and replace with numbered placeholders
            numbers = re.findall(r"[\d.]+", message)   # e.g., ['1.0', '0.36']
            protected = re.sub(r"[\d.]+", lambda m: f"<NUM{numbers.index(m.group())}>", message)

            # Step 2: Translate only the text part
            translated = hindi_translator(protected)[0]['translation_text']

            # Step 3: Restore original numbers
            for i, num in enumerate(numbers):
                translated = translated.replace(f"<NUM{i}>", num)

            return translated

        except Exception as e:
            return f"[Hindi] translation unavailable: {message} (error: {e})"

    elif target_lang == "Assamese":
        return f"[Assamese translation unavailable] {message}"
