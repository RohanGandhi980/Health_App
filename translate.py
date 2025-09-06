from transformers import MarianMTModel, MarianTokenizer

def multilingual_alert(message: str, target_lang: str) -> str:
    try:
        model_name_map = {
            "English": "Helsinki-NLP/opus-mt-en-en",   # dummy passthrough
            "Hindi": "Helsinki-NLP/opus-mt-en-hi",
            "Assamese": "Helsinki-NLP/opus-mt-en-as"
        }

        model_name = model_name_map.get(target_lang, "Helsinki-NLP/opus-mt-en-en")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs, max_length=256)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    except Exception as e:
        return f"[{target_lang}] translation unavailable: {message} (error: {e})"
