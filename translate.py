# translate.py
from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "facebook/nllb-200-distilled-600M"

LANG_TAGS = {
    "English": "eng_Latn",   
    "Hindi": "hin_Deva",
    "Assamese": "asm_Beng",
}

@lru_cache(maxsize=1)
def _load_nllb():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    return tokenizer, model, device

def multilingual_alert(message: str, target_lang: str) -> str:
    """
    Translate an English message into Hindi/Assamese using NLLB.
    English returns the original message.
    """
    if target_lang == "English":
        return message

    try:
        tokenizer, model, device = _load_nllb()

        tokenizer.src_lang = LANG_TAGS["English"]
        tgt_lang = LANG_TAGS[target_lang]

        inputs = tokenizer(message, return_tensors="pt").to(device)

        forced_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_id,
            max_length=256,
            num_beams=4,
        )

        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    except Exception as e:
        return f"[{target_lang} translation unavailable] {message}  (error: {e})"
