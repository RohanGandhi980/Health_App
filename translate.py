LANG_MODELS = {
    "Hindi": "Helsinki-NLP/opus-mt-en-hi"
    # Assamese removed because no model exists
}

def multilingual_alert(message: str, target_lang: str) -> str:
    if target_lang == "English":
        return message
    if target_lang == "Assamese":
        return f"[Assamese translation unavailable] {message}"

    try:
        model_name = LANG_MODELS[target_lang]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        inputs = tokenizer(message, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    except Exception as e:
        return f"[{target_lang}] translation unavailable: {message} (error: {e})"
