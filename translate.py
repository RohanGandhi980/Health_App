from transformers import MarianMTModel, MarianTokenizer

MODEL_MAP = {
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Assamese": "Helsinki-NLP/opus-mt-en-as"
}

loaded_models = {}

def multilingual_alert(message: str, target_lang: str) -> str:
    """
    Translate the message into the target language (Hindi, Assamese, or English).
    Falls back to the original message if translation fails.
    """
    if target_lang == "English":
        return message  

    try:
        if target_lang not in loaded_models:
            model_name = MODEL_MAP[target_lang]
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            loaded_models[target_lang] = (tokenizer, model)

        tokenizer, model = loaded_models[target_lang]

        inputs = tokenizer([message], return_tensors="pt", padding=True)
        translated = model.generate(**inputs, max_length=256, num_beams=4)
        output = tokenizer.decode(translated[0], skip_special_tokens=True)

        return output

    except Exception as e:
        return f"[{target_lang} translation unavailable] {message} (error: {e})"
