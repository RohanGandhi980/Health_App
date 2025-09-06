# translate.py
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HI_MODEL_ID = "Helsinki-NLP/opus-mt-en-hi"

@lru_cache(maxsize=1)
def _load_hi():
    """
    Lazy-load and cache the English→Hindi model once per process.
    """
    tokenizer = AutoTokenizer.from_pretrained(HI_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(HI_MODEL_ID)
    return tokenizer, model

def multilingual_alert(message: str, target_lang: str) -> str:
    """
    English → return as-is
    Hindi   → translate with Helsinki-NLP/opus-mt-en-hi
    Any other language → return as-is (temporary)
    """
    if target_lang == "English":
        return message
    if target_lang != "Hindi":
        # Assamese (and others) temporarily disabled
        return message

    try:
        tokenizer, model = _load_hi()
        inputs = tokenizer(message, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=128, num_beams=4)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Hindi translation unavailable] {message} (error: {e})"
